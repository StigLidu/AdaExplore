import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configurations exploring larger BLOCK sizes and warp configs for Ampere GPUs (A6000).
# Larger BLOCK processes more spatial elements per block, reducing loop overhead for very large N.
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 4096},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 8192},  num_warps=8, num_stages=2),
    triton.Config({"BLOCK": 16384}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK": 32768}, num_warps=8, num_stages=2),
]


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['N'])
@triton.jit
def _fused_clamp_softmax_scale_kernel(
    inp_ptr,       # pointer to input flattened matrix (rows x N)
    out_ptr,       # pointer to output flattened matrix (rows x N)
    scale_ptr,     # pointer to per-row scale scalar array (rows,)
    rows,          # number of rows (batch * channels)
    N,             # number of columns (spatial flattened length)
    clamp_min,     # clamp min (runtime float)
    clamp_max,     # clamp max (runtime float)
    BLOCK: tl.constexpr,  # columns processed per block
):
    """
    Fused kernel: clamp -> numerically-stable softmax (over each row) -> multiply by per-row scale.
    Uses a streaming log-sum-exp reduction to compute max and denom in a single pass,
    then a second pass to write normalized outputs multiplied by scale.

    Each program handles one row (program_id(0) indexes rows).
    BLOCK is the chunk size in the flattened spatial dimension processed per loop iteration.
    """
    row = tl.program_id(0)
    # Row pointer offset (in elements)
    row_offset = row * N

    # Very negative value for masked/out-of-bounds elements so they don't affect max/sum
    neg_inf = -1e20

    # Running maximum and running sum for log-sum-exp streaming
    m = neg_inf
    s = 0.0

    offs = tl.arange(0, BLOCK)

    # First pass: streaming log-sum-exp with clamping applied to valid entries
    start = 0
    # iterate in Python over ranges of BLOCK (this expands to loop in kernel)
    for start in range(0, N, BLOCK):
        idx = start + offs
        mask = idx < N

        # Load block (masked). Use neg_inf for masked elements.
        x = tl.load(inp_ptr + row_offset + idx, mask=mask, other=neg_inf)

        # Apply clamp only for valid elements; masked ones remain neg_inf
        # Use tl.where with mask to avoid clamping neg_inf values
        x_clamped = tl.where(mask, tl.minimum(tl.maximum(x, clamp_min), clamp_max), neg_inf)

        # Block max and sum
        block_max = tl.max(x_clamped, axis=0)
        # Update streaming running max and denom in a numerically stable manner
        new_m = tl.maximum(m, block_max)
        s = s * tl.exp(m - new_m) + tl.sum(tl.exp(x_clamped - new_m), axis=0)
        m = new_m

    denom = s

    # Load per-row scale
    scale_val = tl.load(scale_ptr + row)

    # Second pass: write normalized softmax * scale
    for start in range(0, N, BLOCK):
        idx = start + offs
        mask = idx < N

        x = tl.load(inp_ptr + row_offset + idx, mask=mask, other=neg_inf)
        x_clamped = tl.where(mask, tl.minimum(tl.maximum(x, clamp_min), clamp_max), neg_inf)

        # For masked positions, exp(-inf)=0 so result is 0.
        out_val = tl.exp(x_clamped - m) / denom
        out_val = out_val * scale_val
        tl.store(out_ptr + row_offset + idx, out_val, mask=mask)


def triton_fused_clamp_softmax_scale(x: torch.Tensor, scale: torch.Tensor, clamp_min: float, clamp_max: float):
    """
    x: tensor of shape (B, C, D, H, W), float32, CUDA
    scale: tensor of shape (1, C, 1, 1, 1) or (C,) or broadcastable to (B, C, 1, 1, 1)
    Returns: tensor with same shape where clamp -> softmax (over flattened spatial dims per (B,C)) -> scale applied.
    """
    assert x.is_cuda, "Input must be on CUDA"
    B, C, D, H, W = x.shape
    rows = B * C
    N = D * H * W

    # Flatten spatial dims -> (rows, N) and ensure contiguous for strided access in Triton
    x_flat = x.view(rows, N).contiguous()
    out_flat = torch.empty_like(x_flat)

    # Prepare per-row scale: expand channel scale across batch then flatten
    # Accept scale shapes like (1,C,1,1,1) or (C,)
    if scale.ndim == 5:
        scale_c = scale.view(-1)
    else:
        scale_c = scale.view(-1)
    # Expand across batch and flatten to rows
    scale_rows = scale_c.unsqueeze(0).expand(B, C).contiguous().view(rows).contiguous()

    # Grid: one program per row
    grid = lambda meta: (rows,)

    # Launch kernel (autotuned BLOCK)
    _fused_clamp_softmax_scale_kernel[grid](x_flat, out_flat, scale_rows, rows, N, float(clamp_min), float(clamp_max))

    # Reshape back to (B, C, D, H, W)
    out = out_flat.view(B, C, D, H, W)
    return out


class ModelNew(nn.Module):
    """
    Optimized model:
      - Keep nn.AvgPool3d and nn.ConvTranspose3d (PyTorch kernels are highly optimized).
      - Fuse clamp + spatial softmax + per-channel scale into a single Triton kernel
        to reduce memory traffic and improve throughput for large spatial sizes.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.avg_pool = nn.AvgPool3d(pool_kernel_size)
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        # learnable scale per out_channel, same shape as original for compatibility
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))

    def forward(self, x):
        """
        x: (B, in_channels, D, H, W)
        """
        x = self.avg_pool(x)
        x = self.conv_transpose(x)
        # fused clamp + softmax (over flattened spatial dims per (B,C)) + scale
        x = triton_fused_clamp_softmax_scale(x, self.scale, self.clamp_min, self.clamp_max)
        return x