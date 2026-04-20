import math
import torch
import torch.nn as nn
import torch.nn.init as init
import triton
import triton.language as tl

# Autotune configs tuned for NVIDIA A6000 (Ampere). These explore larger N tiling
# and different M/K tilings to maximize tensor-core utilization and occupancy.
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32},  num_warps=4,  num_stages=2),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 256, "BLOCK_K": 32},  num_warps=8,  num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32},  num_warps=8,  num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 512, "BLOCK_K": 32},  num_warps=8,  num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 512, "BLOCK_K": 32},  num_warps=8,  num_stages=3),
]

@triton.autotune(
    configs=AUTOTUNE_CONFIGS,
    key=['B', 'N', 'K']
)
@triton.jit
def _fused_gemm_mul_leaky_kernel(
    x_ptr,          # pointer to X (B, K)  -- expects fp16 pointer
    w_ptr,          # pointer to W (N, K)  -- expects fp16 pointer
    b_ptr,          # pointer to bias (N,)  -- fp32 pointer
    out_ptr,        # pointer to output (B, N) -- fp32 pointer
    B,              # batch (M)
    N,              # output features
    K,              # input features
    multiplier,     # scalar multiplier (float)
    negative_slope, # leaky relu negative slope (float)
    stride_x_row,   # row stride for X (in elements)
    stride_w_row,   # row stride for W (in elements)
    stride_out_row, # row stride for output (in elements)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Block indices
    bm = tl.program_id(0)
    bn = tl.program_id(1)

    # Offsets within the matrices for this block
    m_offs = bm * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = bn * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offs = tl.arange(0, BLOCK_K)

    # Masks for boundaries
    mask_m = m_offs < B
    mask_n = n_offs < N

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks; we try to process two BLOCK_K chunks per loop iteration
    k_ptr = 0
    while k_ptr < K:
        # First chunk
        k_chunk0 = k_ptr + k_offs
        mask_k0 = k_chunk0 < K

        x_addr0 = (m_offs[:, None] * stride_x_row) + k_chunk0[None, :]
        w_addr0 = (n_offs[:, None] * stride_w_row) + k_chunk0[None, :]

        # Load fp16 tiles; other=0.0 for masked lanes
        x_tile0 = tl.load(x_ptr + x_addr0, mask=(mask_m[:, None] & mask_k0[None, :]), other=0.0)
        w_tile0 = tl.load(w_ptr + w_addr0, mask=(mask_n[:, None] & mask_k0[None, :]), other=0.0)

        # Use tensor cores: fp16 inputs -> fp32 accumulation
        acc += tl.dot(x_tile0, w_tile0.T)

        k_ptr += BLOCK_K

        # Second chunk (if any) to increase arithmetic intensity and hide memory latency
        if k_ptr < K:
            k_chunk1 = k_ptr + k_offs
            mask_k1 = k_chunk1 < K

            x_addr1 = (m_offs[:, None] * stride_x_row) + k_chunk1[None, :]
            w_addr1 = (n_offs[:, None] * stride_w_row) + k_chunk1[None, :]

            x_tile1 = tl.load(x_ptr + x_addr1, mask=(mask_m[:, None] & mask_k1[None, :]), other=0.0)
            w_tile1 = tl.load(w_ptr + w_addr1, mask=(mask_n[:, None] & mask_k1[None, :]), other=0.0)

            acc += tl.dot(x_tile1, w_tile1.T)

            k_ptr += BLOCK_K

    # Add bias (fp32), multiply and apply LeakyReLU
    b_vals = tl.load(b_ptr + n_offs, mask=mask_n, other=0.0)  # (BLOCK_N,)

    out_tile = acc + b_vals[None, :]     # broadcast bias over rows
    out_tile = out_tile * multiplier

    # LeakyReLU: x if x>0 else x * negative_slope
    pos = tl.maximum(out_tile, 0.0)
    neg = tl.minimum(out_tile, 0.0) * negative_slope
    out_tile = pos + neg

    # Store results
    out_addr = (m_offs[:, None] * stride_out_row) + n_offs[None, :]
    mask_store = (mask_m[:, None] & mask_n[None, :])
    tl.store(out_ptr + out_addr, out_tile, mask=mask_store)


def triton_fused_gemm_mul_leaky(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, multiplier: float, negative_slope: float):
    """
    Wrapper to launch the autotuned Triton kernel.
    Converts inputs to fp16 device views to leverage tensor cores for high throughput.
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "All tensors must be on CUDA."
    assert x.dtype == torch.float32 and weight.dtype == torch.float32 and bias.dtype == torch.float32

    B, K = x.shape
    N, K_w = weight.shape
    assert K == K_w, "K dimensions must match"

    # Contiguous device views and convert to fp16 for kernel loads
    x_contig = x.contiguous()
    w_contig = weight.contiguous()
    b_contig = bias.contiguous()

    # Use half precision views for input matrices to reduce bandwidth and engage tensor cores
    x_half = x_contig.half()
    w_half = w_contig.half()

    out = torch.empty((B, N), device=x.device, dtype=torch.float32)

    # Strides in elements (row-major contiguous)
    stride_x_row = x_half.stride(0)
    stride_w_row = w_half.stride(0)
    stride_out_row = out.stride(0)

    # Grid for autotuning
    grid = lambda meta: (
        (B + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
        (N + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],
    )

    # Launch kernel; autotune selects best BLOCK sizes and resources
    _fused_gemm_mul_leaky_kernel[grid](
        x_half, w_half, b_contig, out,
        B, N, K,
        float(multiplier),
        float(negative_slope),
        stride_x_row, stride_w_row, stride_out_row
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized Model that fuses GEMM, scaling, and LeakyReLU in a single Triton kernel.
    Matches the original Model API: Linear -> multiply -> LeakyReLU.
    """
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.multiplier = float(multiplier)
        self.negative_slope = float(negative_slope)

        # weight: (out_features, in_features) consistent with nn.Linear
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float32))

        # Initialize parameters similarly to nn.Linear
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1.0 / math.sqrt(in_features) if in_features > 0 else 0.0
        init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if not x.is_cuda:
            raise RuntimeError("ModelNew requires CUDA tensors as input.")
        x = x.contiguous()
        return triton_fused_gemm_mul_leaky(x, self.weight, self.bias, self.multiplier, self.negative_slope)