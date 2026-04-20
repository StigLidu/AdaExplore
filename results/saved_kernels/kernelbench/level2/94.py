import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Triton GEMM kernel (row-major A, B, C)
@triton.jit
def _triton_gemm_kernel(
    A_ptr, B_ptr, C_ptr, BIAS_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_bias,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # pointers to the beginning of the a and b blocks
    a_ptrs = A_ptr + offs_m[:, None] * stride_am + tl.arange(0, BLOCK_K)[None, :] * stride_ak
    b_ptrs = B_ptr + tl.arange(0, BLOCK_K)[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Double-buffered preload: load first tile
    rk = tl.arange(0, BLOCK_K)
    k0 = 0
    mask_a = (offs_m[:, None] < M) & ((k0 + rk[None, :]) < K)
    mask_b = ((k0 + rk[:, None]) < K) & (offs_n[None, :] < N)

    a = tl.load(a_ptrs + k0 * stride_ak, mask=mask_a, other=0.0)
    b = tl.load(b_ptrs + k0 * stride_bk, mask=mask_b, other=0.0)
    # Cast tiles to FP16 for compute to utilize Tensor Cores; accumulation remains FP32
    a = a.to(tl.float16)
    b = b.to(tl.float16)

    # iterate over remaining K tiles, preloading the next tile while computing current
    for k0 in range(BLOCK_K, K, BLOCK_K):
        next_mask_a = (offs_m[:, None] < M) & ((k0 + rk[None, :]) < K)
        next_mask_b = ((k0 + rk[:, None]) < K) & (offs_n[None, :] < N)

        a_next = tl.load(a_ptrs + k0 * stride_ak, mask=next_mask_a, other=0.0)
        b_next = tl.load(b_ptrs + k0 * stride_bk, mask=next_mask_b, other=0.0)
        a_next = a_next.to(tl.float16)
        b_next = b_next.to(tl.float16)

        # compute on the currently-loaded tile (FP16 dot), cast result to FP32 and accumulate
        acc += tl.dot(a, b).to(tl.float32)

        # advance buffers
        a = a_next
        b = b_next

    # final tile compute
    acc += tl.dot(a, b).to(tl.float32)

    # Load fused bias for outputs (shape: N,), broadcast along M
    bias_vals = tl.load(BIAS_ptr + offs_n * stride_bias, mask=(offs_n < N), other=0.0)  # shape (BLOCK_N,)
    acc += bias_vals[None, :]

    # Apply Hardtanh clamp to [-1.0, 1.0] (fused)
    acc = tl.maximum(acc, -1.0)
    acc = tl.minimum(acc, 1.0)

    # store results
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask_c)


def triton_gemm(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor):
    """
    Compute a @ b where:
      - a: (M, K) row-major
      - b: (N, K) row-major (native weight layout: out_features, in_features)
    Returns C: (M, N) row-major
    Uses Triton kernel with fixed block sizes tuned for Ampere.
    The kernel expects B laid out so that K is the fastest-changing dimension (stride along K == 1),
    and it will add the provided fused bias and apply Hardtanh clamp before storing.
    """
    assert a.is_cuda and b.is_cuda and bias.is_cuda, "Triton GEMM requires CUDA tensors"
    assert a.dtype == torch.float32 and b.dtype == torch.float32 and bias.dtype == torch.float32

    M, K = a.shape
    N, Kb = b.shape
    assert K == Kb, "Incompatible shapes for GEMM (a: MxK, b: N x K expected)"

    # Make contiguous in their native layouts
    a_ = a.contiguous()
    b_ = b.contiguous()      # b_ shape: (N, K) so stride along K is b_.stride(1) (ideally 1)
    bias_ = bias.contiguous()  # shape: (N,)

    # output
    c = torch.empty((M, N), device=a_.device, dtype=torch.float32)

    # block sizes (tuned for better occupancy on A6000/Ampere)
    # smaller M/N tiles allow more concurrent programs; K blocking of 64 is a good tradeoff
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64

    grid = ( (M + BLOCK_M - 1) // BLOCK_M, (N + BLOCK_N - 1) // BLOCK_N )

    _triton_gemm_kernel[grid](
        a_, b_, c, bias_,
        M, N, K,
        a_.stride(0), a_.stride(1),
        b_.stride(1), b_.stride(0),   # stride_bk (along K) is b_.stride(1); stride_bn (along N) is b_.stride(0)
        c.stride(0), c.stride(1),
        bias_.stride(0),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return c


class ModelNew(nn.Module):
    """
    Optimized Model with custom Triton GEMM kernel for the heavy matrix multiply.
    Preserves the original module parameters (Linear and GroupNorm) and behavior.
    The sequence remains:
      x -> Linear (GEMM + linear_bias) -> extra bias -> Hardtanh -> Mish -> GroupNorm
    """

    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super(ModelNew, self).__init__()
        # Keep the original modules so parameters remain registered and usable by optimizers.
        self.gemm = nn.Linear(in_features, out_features)  # has weight and bias
        # extra bias added after gemm
        self.bias = nn.Parameter(torch.randn(bias_shape))
        # GroupNorm (affine=True by default) to match original behavior
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=out_features)
        # hardtanh parameters are fixed defaults in original model; using clamp during forward
        # Mish will be computed via PyTorch ops (on GPU) after Triton GEMM

    def forward(self, x):
        # If not on CUDA or types mismatch, fall back to PyTorch implementation for correctness
        if not x.is_cuda or self.gemm.weight.device != x.device or x.dtype != torch.float32:
            # fall back to reference implementation
            x = self.gemm(x)
            x = x + self.bias
            x = torch.clamp(x, min=self.gemm.bias.min().item() if False else -1.0, max=1.0)  # hardtanh default (-1,1)
            # Mish: x * tanh(softplus(x))
            x = x * torch.tanh(F.softplus(x))
            x = self.groupnorm(x)
            return x

        # Perform GEMM via Triton: x @ weight.T
        # Prepare operands: a = x (M,K), b = weight in native layout (N, K) where K is the fastest-changing dim
        a = x
        # Use weight in its native (out_features, in_features) layout so K (in_features) is contiguous.
        w = self.gemm.weight.contiguous()

        # fuse linear bias and extra bias into a single vector passed to the kernel
        fused_bias = (self.gemm.bias + self.bias).contiguous()

        # Compute matmul on GPU; the Triton kernel will also add fused_bias and apply Hardtanh clamping
        out = triton_gemm(a, w, fused_bias)  # shape (batch, out_features)

        # Mish: x * tanh(softplus(x))
        out = out * torch.tanh(F.softplus(out))

        # GroupNorm (uses module parameters)
        out = self.groupnorm(out)

        return out