import torch
import torch.nn as nn
import triton
import triton.language as tl

# Triton kernel that fuses Swish activation, bias addition, and GroupNorm (process BLOCK_N rows per program)
@triton.jit
def _swish_groupnorm_kernel(
    x_ptr,           # pointer to input x (N, C)
    bias_ptr,        # pointer to added bias (C,)
    gn_w_ptr,        # pointer to groupnorm weight (C,)
    gn_b_ptr,        # pointer to groupnorm bias (C,)
    out_ptr,         # pointer to output (N, C)
    N,               # number of samples
    C,               # number of channels
    G,               # number of groups
    channels_per_group,  # channels per group (C // G)
    eps,             # epsilon for groupnorm
    BLOCK_N: tl.constexpr,  # number of rows per program
    BLOCK_C: tl.constexpr,  # compile-time block size (channels per group)
):
    # program indices: one program per (n_block, group g)
    n_block = tl.program_id(0)
    g = tl.program_id(1)

    # row indices this program handles: [n_block * BLOCK_N, ..., n_block * BLOCK_N + BLOCK_N-1]
    rn = tl.arange(0, BLOCK_N)                     # (BLOCK_N,)
    row_ids = n_block * BLOCK_N + rn               # (BLOCK_N,)

    # channel indices for this group
    c_block = tl.arange(0, BLOCK_C)                # (BLOCK_C,)
    ch_idx = g * channels_per_group + c_block     # (BLOCK_C,)
    mask_c = ch_idx < C

    # build (BLOCK_N, BLOCK_C) offsets and masks
    row_ids_b = row_ids[:, None]                   # (BLOCK_N, 1)
    offs = row_ids_b * C + ch_idx[None, :]         # (BLOCK_N, BLOCK_C)

    mask_row = row_ids < N                         # (BLOCK_N,)
    mask = mask_row[:, None] & mask_c[None, :]     # (BLOCK_N, BLOCK_C)

    # Load inputs and params (broadcast per-channel params across BLOCK_N rows)
    x_vals = tl.load(x_ptr + offs, mask=mask, other=0.0)               # (BLOCK_N, BLOCK_C)
    bias_vals = tl.load(bias_ptr + ch_idx, mask=mask_c, other=0.0)    # (BLOCK_C,)
    gn_w_vals = tl.load(gn_w_ptr + ch_idx, mask=mask_c, other=1.0)
    gn_b_vals = tl.load(gn_b_ptr + ch_idx, mask=mask_c, other=0.0)

    # Swish: x * sigmoid(x)
    sig = 1.0 / (1.0 + tl.exp(-x_vals))
    swish = x_vals * sig

    # add the external bias (broadcasted)
    vals = swish + bias_vals[None, :]   # (BLOCK_N, BLOCK_C)

    # compute sum and sum-of-squares per row in one pass (reduce over channels)
    sum_vals = tl.sum(vals, axis=1)          # (BLOCK_N,)
    sumsq = tl.sum(vals * vals, axis=1)      # (BLOCK_N,)

    M = tl.cast(BLOCK_C, tl.float32)
    mean = sum_vals / M                      # (BLOCK_N,)
    var = sumsq / M - mean * mean           # (BLOCK_N,)

    invstd = 1.0 / tl.sqrt(var + eps)        # (BLOCK_N,)

    # normalize and apply affine params (broadcast channel-wise params)
    normalized = (vals - mean[:, None]) * invstd[:, None]    # (BLOCK_N, BLOCK_C)
    out_vals = normalized * gn_w_vals[None, :] + gn_b_vals[None, :]

    # store output (masked for tails)
    tl.store(out_ptr + offs, out_vals, mask=mask)


def fused_swish_groupnorm(x: torch.Tensor, bias: torch.Tensor, gn: nn.GroupNorm):
    """
    Wrapper that launches the Triton kernel to perform:
      out = GroupNorm(swish(x) + bias)
    where GroupNorm is applied per-sample across groups.

    Inputs:
      x: (N, C) float32 tensor on CUDA
      bias: (C,) float32 tensor on CUDA
      gn: nn.GroupNorm instance (used for weight, bias, eps, and number of groups)
    Returns:
      out: (N, C) float32 tensor on CUDA
    """
    assert x.is_cuda and bias.is_cuda and gn.weight.is_cuda, "All tensors must be on CUDA."
    assert x.dtype == torch.float32, "Expected float32 inputs."

    N, C = x.shape
    G = gn.num_groups
    assert C % G == 0, "Number of channels must be divisible by num_groups."
    channels_per_group = C // G
    # Use channels_per_group as compile-time BLOCK_C
    BLOCK_C = channels_per_group
    # Tile the batch dimension: number of rows each Triton program handles.
    # A moderate default for A6000 is 8; can be tuned (4/8/16).
    BLOCK_N = 8

    x_contig = x.contiguous()
    bias_contig = bias.contiguous()
    gn_w = gn.weight.contiguous()
    gn_b = gn.bias.contiguous()
    out = torch.empty_like(x_contig)

    # grid: one program per (n_block, g)
    grid = ((N + BLOCK_N - 1) // BLOCK_N, G)

    # Launch the kernel. Pass BLOCK_N and BLOCK_C as constexpr kwargs.
    _swish_groupnorm_kernel[grid](
        x_contig, bias_contig, gn_w, gn_b, out,
        N, C, G, channels_per_group, float(gn.eps),
        BLOCK_N=BLOCK_N, BLOCK_C=BLOCK_C
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that keeps the original Linear (matmul) and uses a fused Triton kernel
    to perform Swish activation, add bias, and GroupNorm in one pass for performance.
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        # Keep the Linear layer for GEMM (relying on optimized cuBLAS)
        self.matmul = nn.Linear(in_features, out_features)
        # Additional bias that is added after Swish in the original model
        self.bias = nn.Parameter(torch.randn(bias_shape))
        # Create a GroupNorm module to hold affine parameters and eps
        self.group_norm = nn.GroupNorm(num_groups, out_features)

    def forward(self, x):
        # x: (N, in_features)
        x = self.matmul(x)              # (N, C)
        # Fuse swish + bias + groupnorm in Triton kernel
        x = fused_swish_groupnorm(x, self.bias, self.group_norm)
        return x


# Keep helper functions as in the original module for compatibility
batch_size = 32768
in_features = 1024
out_features = 4096
num_groups = 64
bias_shape = (out_features,)


def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]


def get_init_inputs():
    return [in_features, out_features, num_groups, bias_shape]