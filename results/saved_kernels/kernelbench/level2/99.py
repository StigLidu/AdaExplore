import torch
import torch.nn as nn
import triton
import triton.language as tl

# Aggressive autotune candidates tuned for A6000 (Ampere)
# Favor larger BLOCK_N and BLOCK_K to reduce number of N-block passes for large N=8192.
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 256,  "BLOCK_K": 64},  num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256,  "BLOCK_K": 64},  num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 512,  "BLOCK_K": 64},  num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 512,  "BLOCK_K": 64},  num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 512,  "BLOCK_K": 128}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 512,  "BLOCK_K": 128}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 1024, "BLOCK_K": 128}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 1024, "BLOCK_K": 128}, num_warps=8, num_stages=4),
    # Smaller tiles as fallbacks
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128,  "BLOCK_K": 64},  num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128,  "BLOCK_K": 64},  num_warps=8, num_stages=3),
]


@triton.autotune(
    configs=AUTOTUNE_CONFIGS,
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_bias_gelu_kernel(
    A_ptr,            # A pointer (M, K) - fp16
    B_ptr,            # B pointer (K, N) - fp16 (pass weight.t().half().contiguous())
    C_ptr,            # output pointer (M, N) - fp16
    M, N, K,          # matrix dimensions
    stride_am, stride_ak,  # A strides
    stride_bk, stride_bn,  # B strides
    stride_cm, stride_cn,  # C strides
    bias_ptr, stride_bias, # bias pointer and stride (bias shape N,) bias is fp32
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # program indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N

    offs_m = row_start + tl.arange(0, BLOCK_M)
    offs_n = col_start + tl.arange(0, BLOCK_N)
    k_offs = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # loop over K in tiles
    for k in range(0, K, BLOCK_K):
        k_block = k + k_offs
        mask_a = (offs_m[:, None] < M) & (k_block[None, :] < K)
        mask_b = (k_block[:, None] < K) & (offs_n[None, :] < N)

        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + k_block[None, :] * stride_ak)
        b_ptrs = B_ptr + (k_block[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        A_tile = tl.load(a_ptrs, mask=mask_a, other=0.0)  # fp16 loads
        B_tile = tl.load(b_ptrs, mask=mask_b, other=0.0)  # fp16 loads

        # tl.dot on fp16 inputs yields fp32 accumulation
        acc += tl.dot(A_tile, B_tile)

    # Add bias (broadcast across rows). Load bias in fp32 and broadcast.
    bias_vals = tl.load(bias_ptr + offs_n * stride_bias, mask=offs_n < N, other=0.0)
    acc = acc + bias_vals[None, :]

    # GELU approximation: x * sigmoid(1.702 * x)
    acc = acc * (1.0 / (1.0 + tl.exp(-1.702 * acc)))

    # store as fp16 to reduce memory traffic
    acc_fp16 = acc.to(tl.float16)
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc_fp16, mask=mask_c)


def triton_matmul_bias_gelu(A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor):
    """
    A: (M, K) contiguous, torch.float16
    B: (K, N) contiguous (weight.t().half().contiguous()), torch.float16
    bias: (N,) torch.float32
    returns C: (M, N) in torch.float16
    """
    assert A.is_cuda and B.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    assert A.dtype == torch.float16 and B.dtype == torch.float16 and bias.dtype == torch.float32

    A_ = A.contiguous()
    B_ = B.contiguous()
    bias_ = bias.contiguous()

    M, K = A_.shape
    Kb, N = B_.shape
    assert Kb == K, "K dimension mismatch between A and B"
    C = torch.empty((M, N), device=A_.device, dtype=torch.float16)

    stride_am, stride_ak = A_.stride()
    stride_bk, stride_bn = B_.stride()
    stride_cm, stride_cn = C.stride()
    stride_bias = bias_.stride()[0]

    grid = lambda meta: (
        (M + meta['BLOCK_M'] - 1) // meta['BLOCK_M'],
        (N + meta['BLOCK_N'] - 1) // meta['BLOCK_N'],
    )

    matmul_bias_gelu_kernel[grid](
        A_, B_, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        bias_, stride_bias,
    )
    return C


# Optimized row-wise softmax: each program handles one row, larger BLOCK to reduce passes.
@triton.jit
def row_softmax_kernel(
    X_ptr, Y_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    neg_inf = -1e20

    cur_max = neg_inf
    cur_sum = 0.0

    # First pass: compute row-wise max and sum (stable merge across column blocks)
    for col_start in range(0, N, BLOCK):
        idx = col_start + offs
        mask = idx < N
        ptrs = X_ptr + row * stride_xm + idx * stride_xn
        vals = tl.load(ptrs, mask=mask, other=0.0)      # fp16 load
        vals_f32 = vals.to(tl.float32)

        # block-wise max and sum
        block_max = tl.max(vals_f32, axis=0)
        exp_vals = tl.exp(vals_f32 - block_max)
        block_sum = tl.sum(exp_vals, axis=0)

        # stable merge with current aggregate
        cond = block_max > cur_max
        exp_cur_minus_block = tl.exp(cur_max - block_max)
        exp_block_minus_cur = tl.exp(block_max - cur_max)
        cur_sum = tl.where(cond,
                           cur_sum * exp_cur_minus_block + block_sum,
                           cur_sum + block_sum * exp_block_minus_cur)
        cur_max = tl.where(cond, block_max, cur_max)

    # Second pass: write normalized softmax outputs in fp32
    for col_start in range(0, N, BLOCK):
        idx = col_start + offs
        mask = idx < N
        in_ptrs = X_ptr + row * stride_xm + idx * stride_xn
        out_ptrs = Y_ptr + row * stride_ym + idx * stride_yn
        vals = tl.load(in_ptrs, mask=mask, other=0.0)
        vals_f32 = vals.to(tl.float32)
        res = tl.exp(vals_f32 - cur_max) / cur_sum
        tl.store(out_ptrs, res, mask=mask)


def triton_row_softmax(X: torch.Tensor, BLOCK: int = 1024):
    """
    Compute softmax over dim=1 (rows of shape (M, N)) using Triton.
    X is expected to be fp16 (the GEMM output). Returns fp32 tensor.
    """
    assert X.is_cuda and X.dtype == torch.float16
    X_ = X.contiguous()
    M, N = X_.shape
    Y = torch.empty((M, N), device=X_.device, dtype=torch.float32)

    stride_xm, stride_xn = X_.stride()
    stride_ym, stride_yn = Y.stride()

    grid = (M,)

    # BLOCK is passed as constexpr
    row_softmax_kernel[grid](
        X_, Y,
        M, N,
        stride_xm, stride_xn,
        stride_ym, stride_yn,
        BLOCK,
    )
    return Y


class ModelNew(nn.Module):
    """
    Optimized Model:
      - Fused matmul + bias + GELU implemented in Triton (mixed precision).
      - Row-wise softmax implemented in Triton (numerically stable streaming).
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        # persistent cached half-weight to avoid repeated transposes/casts
        self.register_buffer("_dummy_buf_for_device", torch.empty(0), persistent=False)
        self.W_half = None

    def _ensure_weight_half(self):
        W = self.linear.weight
        # desired shape for B: (K, N) == (in_features, out_features)
        desired_shape = (W.shape[1], W.shape[0])
        if (self.W_half is None) or (self.W_half.shape != desired_shape) or (self.W_half.device != W.device):
            # store transposed half contiguous on the same device as weight
            self.W_half = W.t().half().contiguous().to(W.device)

    def forward(self, x):
        # Move input to CUDA if needed (kernels assume CUDA tensors)
        if not x.is_cuda:
            x = x.cuda()
        W = self.linear.weight
        b = self.linear.bias

        # ensure cached weight in desired layout and dtype
        self._ensure_weight_half()
        W_half = self.W_half

        # cast input to fp16 once per forward
        x_h = x.half().contiguous()

        # bias kept in fp32 for numerical stability
        b_fp32 = b.contiguous()

        # matmul + bias + gelu -> fp16 output
        out_fp16 = triton_matmul_bias_gelu(x_h, W_half, b_fp32)

        # softmax over dim=1 using Triton (returns fp32)
        # Use large BLOCK to reduce passes across N=8192; 1024 works well on Ampere
        out = triton_row_softmax(out_fp16, BLOCK=1024)
        return out


# Keep the original dataset shape variables and helper functions
batch_size = 1024
in_features = 8192
out_features = 8192

def get_inputs():
    # provide CUDA inputs for the optimized kernels
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features]