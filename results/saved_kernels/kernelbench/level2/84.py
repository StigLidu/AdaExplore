import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Autotune configs for matmul kernel
# Expanded to probe larger BLOCK_K and larger square tiles to raise arithmetic intensity,
# and to explore different warp/stage combinations for A6000 (Ampere).
AUTOTUNE_MATMUL = [
    # small tiles (low memory pressure)
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=2),

    # rectangular tiles
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),

    # square tiles with moderate K (better tensor core utilization)
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8, num_stages=4),

    # larger square tiles (reduce launch overhead, better for large M,N)
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=16, num_stages=4),

    # extra variants to give autotune flexibility (different warp/stage tradeoffs)
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=2),
]


@triton.autotune(
    configs=AUTOTUNE_MATMUL,
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_kernel(
    A_ptr,  # pointer to A matrix (M x K)
    B_ptr,  # pointer to B matrix (K x N)
    C_ptr,  # pointer to output matrix (M x N)
    M, N, K,
    stride_am, stride_ak,  # strides for A (along M, along K)
    stride_bk, stride_bn,  # strides for B (along K, along N)
    stride_cm, stride_cn,  # strides for C (along M, along N)
    bias_ptr,               # pointer to bias (N,) (can be 0)
    bias_has,               # int flag (0/1) indicating whether bias_ptr is valid
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Triton GEMM kernel computing C = A @ B + bias
    A: (M, K)
    B: (K, N)
    C: (M, N)
    This kernel tiles MxN output with BLOCK_M x BLOCK_N tiles and reduces over K with BLOCK_K.

    The kernel supports inputs stored as float16 or float32. To reduce memory bandwidth we
    load inputs (which may be float16) and perform the multiply in reduced precision
    (float16) but cast results to float32 and accumulate in float32 for improved numerical
    stability.
    """

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # row and col indices this program will compute
    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # initialize accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # iterate over K in chunks of BLOCK_K
    k = 0
    # hoist the arange for K offsets to avoid re-allocating it every iteration
    arange_k = tl.arange(0, BLOCK_K)
    while k < K:
        # current block size in K (in case K isn't divisible by BLOCK_K)
        k_block = tl.minimum(BLOCK_K, K - k)

        # k indices for this sub-block
        k_offsets = k + arange_k

        # compute addresses for A and B loads
        a_ptrs = A_ptr + (row_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak)
        b_ptrs = B_ptr + (k_offsets[:, None] * stride_bk + col_offsets[None, :] * stride_bn)

        # masks
        mask_a = (row_offsets[:, None] < M) & (k_offsets[None, :] < K)
        mask_b = (k_offsets[:, None] < K) & (col_offsets[None, :] < N)

        # load A and B blocks (BLOCK_M x BLOCK_K) and (BLOCK_K x BLOCK_N)
        A_block = tl.load(a_ptrs, mask=mask_a, other=0.0)
        B_block = tl.load(b_ptrs, mask=mask_b, other=0.0)

        # Mixed precision compute:
        # Cast loaded blocks to float16 for the multiply to reduce memory bandwidth,
        # then cast the product to float32 and accumulate into acc (fp32).
        A_block_f16 = tl.cast(A_block, tl.float16)
        B_block_f16 = tl.cast(B_block, tl.float16)
        prod = tl.dot(A_block_f16, B_block_f16)
        # Ensure accumulation in fp32
        prod_f32 = tl.cast(prod, tl.float32)
        acc += prod_f32

        k += BLOCK_K

    # now write back
    c_ptrs = C_ptr + (row_offsets[:, None] * stride_cm + col_offsets[None, :] * stride_cn)
    mask_c = (row_offsets[:, None] < M) & (col_offsets[None, :] < N)

    # if bias is present, load and add bias (cast bias to fp32 if needed)
    if bias_has != 0:
        bias_vals = tl.load(bias_ptr + col_offsets, mask=(col_offsets < N), other=0.0)
        bias_vals_f32 = tl.cast(bias_vals, tl.float32)
        acc = acc + bias_vals_f32[None, :]

    tl.store(c_ptrs, acc, mask=mask_c)


def triton_gemm(A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    """
    Compute A @ B + bias using the Triton GEMM kernel.

    This function implements a mixed-precision fast path: when inputs are float32,
    we cast A and B (and bias) to float16 to reduce memory traffic; the kernel
    multiplies in reduced precision and accumulates in float32, and the output C
    is produced as float32 so downstream BatchNorm stays in fp32.
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
    assert A.dim() == 2 and B.dim() == 2, "Only 2D tensors supported"
    assert A.shape[1] == B.shape[0], f"K mismatch: {A.shape[1]} vs {B.shape[0]}"

    M, K = A.shape
    Kb, N = B.shape

    # We'll support float32 and float16 inputs. For float32 inputs, use half for
    # transfers into the kernel to reduce memory bandwidth, but produce fp32 output.
    is_fp32_inputs = (A.dtype == torch.float32 and B.dtype == torch.float32)
    is_fp16_inputs = (A.dtype == torch.float16 and B.dtype == torch.float16)
    assert is_fp32_inputs or is_fp16_inputs, "Only float32 or float16 inputs supported"

    # Make contiguous copies of the tensors we will pass to the kernel.
    if is_fp32_inputs:
        A_ = A.contiguous().half()     # cast inputs to fp16 for reduced memory use
        B_ = B.contiguous().half()
        if bias is not None:
            bias_t = bias.contiguous().half()
        else:
            bias_t = None
        # Keep output in fp32 for BatchNorm and numerical stability
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    else:
        # inputs are already fp16 - keep as-is, output still fp32 (accumulation in kernel is fp32)
        A_ = A.contiguous()
        B_ = B.contiguous()
        if bias is not None:
            bias_t = bias.contiguous()
        else:
            bias_t = None
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

    # strides for the actual tensors passed to the kernel
    stride_am, stride_ak = A_.stride()
    stride_bk, stride_bn = B_.stride()
    stride_cm, stride_cn = C.stride()

    # bias handling
    if bias_t is None:
        bias_ptr = 0
        bias_has = 0
    else:
        assert bias_t.dim() == 1 and bias_t.shape[0] == N
        bias_ptr = bias_t
        bias_has = 1

    # choose grid
    def grid(meta):
        # pad M and N up to a modest alignment so most tiles are full (reduces masked boundary work).
        # Use an alignment that works well with common tile sizes (e.g., 128). The kernel uses masks so padding is safe.
        align = 128
        M_padded = ((M + align - 1) // align) * align
        N_padded = ((N + align - 1) // align) * align
        return ( (M_padded + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
                 (N_padded + meta["BLOCK_N"] - 1) // meta["BLOCK_N"], )

    # Launch kernel with the (possibly half) input tensors and fp32 output
    _matmul_kernel[grid](
        A_, B_, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        bias_ptr if bias_has else 0,
        bias_has,
    )

    # C is fp32 as required for BatchNorm; return it
    return C


class ModelNew(nn.Module):
    """
    Optimized Model that replaces the Linear (GEMM) with a Triton-based GEMM kernel,
    and keeps BatchNorm, scaling and Softmax using PyTorch for correctness and ease.
    """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        super(ModelNew, self).__init__()
        # Replace nn.Linear with explicit weight and bias parameters
        self.in_features = in_features
        self.out_features = out_features
        # weight shape matches nn.Linear: (out_features, in_features)
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float32))
        # initialize similarly to nn.Linear (Kaiming uniform)
        bound = 1.0 / math.sqrt(in_features)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

        # Keep BatchNorm1d as before (operates on channels == out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)

        # scale parameter
        self.scale = nn.Parameter(torch.ones(scale_shape, dtype=torch.float32))

        # softmax dimension remains feature dimension (dim=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
          1) GEMM via Triton: x @ weight.T + bias
          2) BatchNorm1d across batch for each feature (PyTorch)
          3) scaling (elementwise multiplication with self.scale)
          4) softmax across features (dim=1)
        """
        # ensure inputs are CUDA float32 contiguous for Triton kernel
        assert x.is_cuda, "Input must be on CUDA"
        assert x.dtype == torch.float32, "Input must be float32"

        # Compute GEMM: x (N, K) @ weight.T (K, M) -> (N, M)
        # Prepare B as weight.T so shape matches (K, N_out)
        weight_t = self.weight.t().contiguous()  # shape (in_features, out_features)
        bias_contig = self.bias.contiguous()

        gemm_out = triton_gemm(x.contiguous(), weight_t, bias_contig)

        # Now use PyTorch BatchNorm (expects (N, C))
        bn_out = self.bn(gemm_out)

        # apply scaling (broadcasting)
        scaled = bn_out * self.scale

        # softmax across features (dim=1)
        out = self.softmax(scaled)
        return out


# Helper functions to match the original module expectations (optional)
# These can be used by external harnesses to create inputs for the model.

batch_size = 1024
in_features = 8192
out_features = 8192
bn_eps = 1e-5
bn_momentum = 0.1
scale_shape = (1,)


def get_inputs():
    return [torch.rand(batch_size, in_features, device='cuda', dtype=torch.float32)]


def get_init_inputs():
    return [in_features, out_features, bn_eps, bn_momentum, scale_shape]