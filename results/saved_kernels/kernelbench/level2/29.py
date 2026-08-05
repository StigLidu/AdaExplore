import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configurations tuned for NVIDIA A6000 (Ampere).
# Expanded pool to include larger BLOCK_N and BLOCK_K candidates (e.g., 1024, 128) which are
# often beneficial for wide N (8192) workloads on Ampere GPUs.
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 1024, "BLOCK_K": 128}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 1024, "BLOCK_K": 128}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 512, "BLOCK_K": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 512, "BLOCK_K": 128}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 512, "BLOCK_N": 512, "BLOCK_K": 128}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 512, "BLOCK_K": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2),
]

@triton.autotune(
    configs=AUTOTUNE_CONFIGS,
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_bias_kernel(
    a_ptr,            # (M, K) half
    b_ptr,            # (K, N) half (transposed weight: K x N)
    bias_ptr,         # (N,) float32
    c_ptr,             # (M, N) float32 out
    M, N, K,
    stride_a_m, stride_a_k,
    stride_b_k, stride_b_n,
    stride_c_m, stride_c_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # tile row/col indices (int32 by default) and caches for address arithmetic
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # cast row/col indices to int64 once (hoisted) for address computations
    rm_i64 = rm.to(tl.int64)
    rn_i64 = rn.to(tl.int64)

    # base offsets (vector) used in K-loop (hoisted to avoid repeated multiplication/casts)
    base_a = rm_i64 * stride_a_m  # shape (BLOCK_M,)
    base_c = rm_i64 * stride_c_m  # shape (BLOCK_M,)
    base_bn = rn_i64 * stride_b_n  # shape (BLOCK_N,)

    # FP32 accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # iterate over K in blocks to compute matmul (fp16 inputs -> fp32 accumulation via tl.dot)
    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        rk_i64 = rk.to(tl.int64)

        # addresses for A (rm x rk) and B (rk x rn)
        # compute addresses by reusing hoisted base offsets
        a_addrs = a_ptr + (base_a[:, None] + rk_i64[None, :] * stride_a_k)
        b_addrs = b_ptr + (rk_i64[:, None] * stride_b_k + base_bn[None, :])

        a_mask = (rm[:, None] < M) & (rk[None, :] < K)
        b_mask = (rk[:, None] < K) & (rn[None, :] < N)

        # load half blocks (keeps as fp16 so tl.dot can use tensor cores with fp32 accumulation)
        a_block = tl.load(a_addrs, mask=a_mask, other=0.0)
        b_block = tl.load(b_addrs, mask=b_mask, other=0.0)

        # accumulate using mixed precision dot
        acc += tl.dot(a_block, b_block)

    # add bias (broadcast across rows). Use rn (int32) to build mask and rn_i64 for addresses.
    bias_addrs = bias_ptr + rn_i64
    bias_vals = tl.load(bias_addrs, mask=rn < N, other=0.0)  # float32
    acc = acc + bias_vals[None, :]

    # Apply Mish twice on the fp32 accumulator before writing to memory (fused activation).
    # Mish(x) = x * tanh(softplus(x)), softplus = log(1+exp(x)). Use clamping to keep exp stable.
    # x is (BLOCK_M, BLOCK_N)
    # first pass
    x1 = acc
    x1_clamped = tl.where(x1 > 50.0, 50.0, tl.where(x1 < -50.0, -50.0, x1))
    u1 = tl.exp(x1_clamped)
    A1 = 1.0 + u1
    A12 = A1 * A1
    t1 = (A12 - 1.0) / (A12 + 1.0)
    m1 = x1 * t1

    # second pass
    x2 = m1
    x2_clamped = tl.where(x2 > 50.0, 50.0, tl.where(x2 < -50.0, -50.0, x2))
    u2 = tl.exp(x2_clamped)
    A2 = 1.0 + u2
    A22 = A2 * A2
    t2 = (A22 - 1.0) / (A22 + 1.0)
    out_block = x2 * t2

    # store result (use same mask to ensure tails remain correct)
    c_addrs = c_ptr + (base_c[:, None] + rn_i64[None, :] * stride_c_n)
    out_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_addrs, out_block, mask=out_mask)





def triton_linear_double_mish(x: torch.Tensor, weight_t_half: torch.Tensor, bias: torch.Tensor):
    """
    Wrapper that launches a MatMul+Bias Triton kernel followed by a small Mish elementwise kernel.
    x: (M, K) half precision CUDA tensor
    weight_t_half: (K, N) half precision CUDA tensor (pre-transposed)
    bias: (N,) float32 CUDA tensor
    returns: (M, N) float32 tensor
    """
    assert x.is_cuda and weight_t_half.is_cuda and bias.is_cuda, "All tensors must be on CUDA"
    assert x.dtype == torch.half and weight_t_half.dtype == torch.half and bias.dtype == torch.float32

    x_h = x.contiguous()
    bias_cont = bias.contiguous()

    M = x_h.shape[0]
    K = x_h.shape[1]
    assert weight_t_half.shape[0] == K, "weight_t_half first dim must equal K"
    N = weight_t_half.shape[1]

    out = torch.empty((M, N), device=x.device, dtype=torch.float32)

    stride_a_m, stride_a_k = x_h.stride(0), x_h.stride(1)
    stride_b_k, stride_b_n = weight_t_half.stride(0), weight_t_half.stride(1)
    stride_c_m, stride_c_n = out.stride(0), out.stride(1)

    # grid for matmul kernel depends on constexpr tile sizes selected by autotuner at runtime
    grid = lambda meta: (
        (M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
        (N + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],
    )

    # launch matmul+bias kernel (autotune selects the best config)
    _matmul_bias_kernel[grid](
        x_h, weight_t_half, bias_cont, out,
        M, N, K,
        stride_a_m, stride_a_k,
        stride_b_k, stride_b_n,
        stride_c_m, stride_c_n
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized model: MatMul+Bias (Triton) + Mish + Mish (Triton).
    - Weight is stored in float32 (nn.Parameter) but a persistent pre-transposed half buffer is kept for fast GEMM.
    - Input is converted to half into a persistent buffer to avoid repeated allocations.
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # weight and bias in fp32 like nn.Linear
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float32))
        # initialize like nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(self.bias, -bound, bound)

        # persistent buffers to avoid repeated allocation / conversion
        # pre-transposed half-precision weight buffer (K, N)
        self.register_buffer("weight_t_half", None)
        # reusable half-precision input buffer (M, K)
        self.register_buffer("input_h", None)

    def forward(self, x):
        # CPU fallback uses standard PyTorch ops for correctness
        if not x.is_cuda:
            return torch.nn.functional.mish(torch.nn.functional.mish(torch.nn.functional.linear(x, self.weight, self.bias)))

        # Ensure pre-transposed half weight exists and is on same device
        expected_w_shape = (self.weight.shape[1], self.weight.shape[0])  # (K, N)
        if (self.weight_t_half is None) or (self.weight_t_half.shape != expected_w_shape) or (self.weight_t_half.device != self.weight.device):
            # create persistent pre-transposed half buffer
            with torch.no_grad():
                self.weight_t_half = self.weight.detach().t().half().contiguous().to(self.weight.device)

        # Prepare half input buffer (reuse if possible)
        x_cont = x.contiguous()
        expected_x_shape = (x_cont.shape[0], x_cont.shape[1])
        if x_cont.dtype == torch.half:
            x_h = x_cont
        else:
            if (self.input_h is None) or (self.input_h.shape != expected_x_shape) or (self.input_h.device != x_cont.device):
                with torch.no_grad():
                    self.input_h = torch.empty(expected_x_shape, dtype=torch.half, device=x_cont.device)
            # copy into preallocated half buffer
            self.input_h.copy_(x_cont.half())
            x_h = self.input_h

        # Launch the matmul+bias Triton kernel then the small Mish kernel.
        out = triton_linear_double_mish(x_h, self.weight_t_half, self.bias)
        return out


# Keep the same helper functions / constants for the benchmarking harness
batch_size = 1024
in_features = 8192
out_features = 8192

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]