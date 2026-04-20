import torch
import torch.nn as nn
import triton
import triton.language as tl

# Triton GEMM kernel (A @ B + bias) where:
# A: (M, K) row-major
# B: (K, N) row-major (we will pass weight.t().contiguous() so B is K x N)
# C: (M, N) row-major (output)
@triton.jit
def _matmul_kernel(
    A_ptr,
    B_ptr,
    bias_ptr,
    C_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)

    row_start = row_block * BLOCK_M
    col_start = col_block * BLOCK_N

    offs_m = row_start + tl.arange(0, BLOCK_M)
    offs_n = col_start + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_b = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        a_ptrs = A_ptr + (offs_m[:, None] * K + offs_k[None, :])
        b_ptrs = B_ptr + (offs_k[:, None] * N + offs_n[None, :])

        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)

        # accumulate along K
        prod = a[:, :, None] * b[None, :, :]
        acc += tl.sum(prod, 1)

    # add bias (per-column)
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    bias_vec = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)  # (BLOCK_N,)
    acc = acc + bias_vec[None, :]

    C_ptrs = C_ptr + (offs_m[:, None] * N + offs_n[None, :])
    tl.store(C_ptrs, acc, mask=mask_c)


# Triton kernel that fuses GroupNorm (per-group), affine, and
# swish -> multiply -> swish for each sample-group block.
@triton.jit
def _groupnorm_swish_mul_swish_kernel(
    X_ptr,           # pointer to input/output X (M * N)
    gamma_ptr,       # groupnorm weight (N,)
    beta_ptr,        # groupnorm bias (N,)
    mult_ptr,        # multiply_weight (N,)
    OUT_ptr,         # pointer to output (M * N) (can be same as X_ptr)
    M, N, G, eps,
    BLOCK_C: tl.constexpr,   # group size (channels per group)
):
    # program_id(0): row index (sample)
    # program_id(1): group index
    row = tl.program_id(0)
    group = tl.program_id(1)

    # base channel for this group
    c_start = group * BLOCK_C
    offs_c = tl.arange(0, BLOCK_C)
    idxs = c_start + offs_c  # channel indices

    # compute flattened offsets for this row and group
    base = row * N + c_start
    offs = base + offs_c
    mask = idxs < N  # in practice N divisible by G so all true

    # load values for this sample-group
    x = tl.load(X_ptr + offs, mask=mask, other=0.0)  # (BLOCK_C,)

    # compute mean and variance over group channels
    # convert to float32 accumulation
    sum_x = tl.sum(x, 0)
    mean = sum_x / BLOCK_C

    diff = x - mean
    sum_sq = tl.sum(diff * diff, 0)
    var = sum_sq / BLOCK_C
    invstd = 1.0 / tl.sqrt(var + eps)

    # load affine params gamma and beta and mult weight
    gamma = tl.load(gamma_ptr + idxs, mask=mask, other=1.0)
    beta = tl.load(beta_ptr + idxs, mask=mask, other=0.0)
    mult = tl.load(mult_ptr + idxs, mask=mask, other=1.0)

    # normalize and apply affine
    x_norm = (x - mean) * invstd
    y = x_norm * gamma + beta

    # first swish: y * sigmoid(y)
    sigmoid_y = 1.0 / (1.0 + tl.exp(-y))
    tmp = y * sigmoid_y

    # multiply by per-channel weight
    tmp = tmp * mult

    # second swish
    sigmoid_tmp = 1.0 / (1.0 + tl.exp(-tmp))
    out = tmp * sigmoid_tmp

    # store result
    tl.store(OUT_ptr + offs, out, mask=mask)


def triton_gemm(A: torch.Tensor, W: torch.Tensor, bias: torch.Tensor):
    assert A.is_cuda and W.is_cuda
    assert A.dtype == torch.float32 and W.dtype == torch.float32
    M, K = A.shape
    N = W.shape[0]  # W is (N, K)
    A_contig = A.contiguous()
    B_t = W.t().contiguous()
    if bias is None:
        bias_tensor = torch.zeros((N,), device=A.device, dtype=torch.float32)
    else:
        bias_tensor = bias.contiguous()

    C = torch.empty((M, N), device=A.device, dtype=torch.float32)

    # tuned block sizes for Ampere
    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 32

    grid = ( (M + BLOCK_M - 1) // BLOCK_M, (N + BLOCK_N - 1) // BLOCK_N )

    _matmul_kernel[grid](
        A_contig, B_t, bias_tensor, C,
        M, N, K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return C


def triton_groupnorm_swish_mul_swish(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, mult: torch.Tensor, num_groups: int, eps: float):
    """
    x: (M, N) tensor (output of GEMM) contiguous
    gamma, beta, mult: (N,) each
    Applies GroupNorm (num_groups) with affine gamma,beta followed by
    swish -> multiply by mult -> swish, all fused in one kernel.
    """
    assert x.is_cuda and gamma.is_cuda and beta.is_cuda and mult.is_cuda
    M, N = x.shape
    assert N % num_groups == 0, "num_groups must divide num_channels"
    G = num_groups
    group_size = N // G
    OUT = torch.empty_like(x)
    BLOCK_C = group_size  # constexpr

    grid = (M, G)
    _groupnorm_swish_mul_swish_kernel[grid](
        x.contiguous(), gamma.contiguous(), beta.contiguous(), mult.contiguous(), OUT,
        M, N, G, eps,
        BLOCK_C=BLOCK_C
    )
    return OUT


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels:
      - Triton GEMM for Linear
      - Triton fused GroupNorm + swish -> multiply -> swish
    GroupNorm affine parameters and multiply_weight are preserved as parameters.
    """
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super(ModelNew, self).__init__()
        # Keep Linear and GroupNorm modules to register parameters and for fallback
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape, dtype=torch.float32))

    def forward(self, x):
        # Expect x on CUDA and float32 for fast path
        if x.is_cuda and x.dtype == torch.float32:
            bias = self.gemm.bias if self.gemm.bias is not None else None
            # GEMM via Triton: A @ W.T + bias
            x_gemm = triton_gemm(x, self.gemm.weight, bias)
            # Fused GroupNorm + activations via Triton
            gamma = self.group_norm.weight if self.group_norm.weight is not None else torch.ones_like(self.multiply_weight)
            beta = self.group_norm.bias if self.group_norm.bias is not None else torch.zeros_like(self.multiply_weight)
            eps = float(self.group_norm.eps)
            out = triton_groupnorm_swish_mul_swish(x_gemm, gamma, beta, self.multiply_weight, self.group_norm.num_groups, eps)
            return out
        else:
            # CPU / non-fp32 fallback to original ops for correctness
            x = self.gemm(x)
            x = self.group_norm(x)
            x = x * torch.sigmoid(x)
            x = x * self.multiply_weight
            x = x * torch.sigmoid(x)
            return x