import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Autotune configurations chosen to favor large tiles and tensor-core-friendly BLOCK_K on Ampere GPUs.
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 512, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=16, num_stages=4),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2),
]

@triton.autotune(
    configs=AUTOTUNE_CONFIGS,
    key=["M", "N", "K"],
)
@triton.jit
def _gemm_mish_kernel(
    A,            # pointer to A: shape (M, K), row-major, fp16
    B,            # pointer to B: shape (K, N), row-major (we pass weight.t().contiguous()), fp16
    C,            # pointer to output C: shape (M, N), row-major, fp32
    M, N, K,      # matrix sizes
    lda, ldb, ldc,# leading dimensions (strides)
    bias,         # bias pointer of shape (N,) or None (fp32)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Mixed-precision GEMM with fused double-Mish activation.
    - A: (M, K) fp16, row-major
    - B: (K, N) fp16, row-major (weight.t().contiguous())
    - C: (M, N) fp32 output, row-major
    Accumulates in fp32.
    After computing the tile accumulation, apply Mish twice: y = mish(mish(x))
    where mish(x) = x * tanh(softplus(x)), computed in a numerically stable manner.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Tile offsets
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    offs_m = m_start + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    offs_n = n_start + tl.arange(0, BLOCK_N)   # [BLOCK_N]

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Load A: address = base + m*lda + k
        a_ptrs = A + (offs_m[:, None] * lda + offs_k[None, :])
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)  # fp16

        # Load B: address = base + k*ldb + n
        b_ptrs = B + (offs_k[:, None] * ldb + offs_n[None, :])
        mask_b = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)  # fp16

        # Prefetch next k-chunk (best-effort)
        next_k = k_start + BLOCK_K
        if next_k < K:
            offs_k_next = next_k + tl.arange(0, BLOCK_K)
            a_next_ptrs = A + (offs_m[:, None] * lda + offs_k_next[None, :])
            b_next_ptrs = B + (offs_k_next[:, None] * ldb + offs_n[None, :])
            _ = tl.load(a_next_ptrs, mask=(offs_m[:, None] < M) & (offs_k_next[None, :] < K), other=0.0)
            _ = tl.load(b_next_ptrs, mask=(offs_k_next[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        # Matrix multiply-accumulate (fp16 inputs, fp32 accumulation)
        acc += tl.dot(a, b)

    # Apply bias if present
    if bias is not None:
        bias_ptrs = bias + offs_n
        mask_bias = offs_n < N
        bias_vals = tl.load(bias_ptrs, mask=mask_bias, other=0.0)  # fp32
        acc = acc + bias_vals[None, :]

    # Now apply Mish twice elementwise on acc (fp32)
    # Define stable softplus: softplus(x) = max(0,x) + log(1 + exp(-abs(x)))
    # Then tanh(softplus(x)) computed with clamping to avoid overflow.
    x = acc  # fp32 tensor [BLOCK_M, BLOCK_N]

    # First Mish
    mpos = tl.maximum(x, 0.0)
    neg_abs = -tl.abs(x)
    sp = mpos + tl.log(1.0 + tl.exp(neg_abs))  # softplus(x) stable
    # Clamp softplus to avoid excessive exp in tanh computation
    sp_clamped = tl.minimum(sp, 20.0)
    e2 = tl.exp(sp_clamped * 2.0)
    tanh_sp = (e2 - 1.0) / (e2 + 1.0)
    mish1 = x * tanh_sp

    # Second Mish (apply to mish1)
    y = mish1
    mpos2 = tl.maximum(y, 0.0)
    neg_abs2 = -tl.abs(y)
    sp2 = mpos2 + tl.log(1.0 + tl.exp(neg_abs2))
    sp2_clamped = tl.minimum(sp2, 20.0)
    e2_2 = tl.exp(sp2_clamped * 2.0)
    tanh_sp2 = (e2_2 - 1.0) / (e2_2 + 1.0)
    mish2 = y * tanh_sp2

    # Store the result
    c_ptrs = C + (offs_m[:, None] * ldc + offs_n[None, :])
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, mish2, mask=out_mask)


def triton_linear_mish(x: torch.Tensor, weight: torch.Tensor = None, bias: torch.Tensor = None, weight_t_h: torch.Tensor = None):
    """
    Wrapper to compute x @ weight.T + bias, followed by Mish applied twice, using a Triton kernel.
    - x: (M, K) fp32 CUDA
    - weight: (N, K) fp32 CUDA (used if weight_t_h not provided)
    - weight_t_h: optional (K, N) fp16 contiguous transposed weight
    returns: (M, N) fp32 CUDA with mish(mish(x @ W^T + b))
    """
    assert x.is_cuda, "x must be on CUDA"
    M, K = x.shape

    if weight_t_h is not None:
        assert weight_t_h.is_cuda and weight_t_h.dtype == torch.float16
        w_t_h = weight_t_h
        K_w, N = w_t_h.shape
        assert K_w == K, "weight_t_h K dimension must match input K"
    else:
        assert weight is not None and weight.is_cuda and weight.dtype == torch.float32
        N = weight.shape[0]
        w_ = weight.contiguous()
        w_t_h = w_.t().contiguous().half()

    # Strides / leading dims
    lda = K
    ldb = N  # B is (K, N) contiguous
    ldc = N  # output row stride

    # Cast input to fp16 (for tensor core efficiency)
    x_ = x.contiguous()
    x_h = x_.half()

    if bias is not None:
        b_ = bias.contiguous()
    else:
        b_ = None

    out = torch.empty((M, N), device=x.device, dtype=torch.float32)

    grid = lambda meta: (
        (M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
        (N + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],
    )

    _gemm_mish_kernel[grid](
        x_h, w_t_h, out,
        M, N, K,
        lda, ldb, ldc,
        b_,
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized Model that fuses the Linear (GEMM) with two Mish activations into a single Triton kernel.
    Caches a transposed fp16 weight to avoid repeated large transpositions/casts.
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        # Keep same parameterization as nn.Linear
        self.linear = nn.Linear(in_features, out_features)
        # Caches to avoid recreating large tensors each forward
        self._w_t_h = None
        self._w_ptr = None
        self._bias_contig = None
        self._bias_ptr = None

    def forward(self, x):
        if not x.is_cuda:
            raise RuntimeError("ModelNew.forward expects input tensor on CUDA")
        x = x.contiguous()
        w = self.linear.weight
        b = self.linear.bias if self.linear.bias is not None else None

        # Update cached transposed fp16 weight if underlying fp32 weight changed.
        w_ptr = w.data_ptr()
        if (self._w_t_h is None) or (self._w_ptr != w_ptr):
            # Create contiguous transposed fp16 copy and cache it.
            self._w_t_h = w.contiguous().half().t().contiguous()
            self._w_ptr = w_ptr

        # Cache contiguous bias if present
        if b is not None:
            b_ptr = b.data_ptr()
            if (self._bias_contig is None) or (self._bias_ptr != b_ptr):
                self._bias_contig = b.contiguous()
                self._bias_ptr = b_ptr
            b_ = self._bias_contig
        else:
            b_ = None

        # Compute fused linear + double-mish via Triton
        out = triton_linear_mish(x, weight_t_h=self._w_t_h, bias=b_)

        return out