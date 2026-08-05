import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configs for different block sizes. Tune to handle a wide range of spatial sizes.
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 128},  num_warps=2, num_stages=2),
    triton.Config({"BLOCK": 256},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 512},  num_warps=8, num_stages=2),
    triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=3),
]


@triton.autotune(
    configs=AUTOTUNE_CONFIGS,
    key=["N"],
)
@triton.jit
def _softmax_clamp_scale_kernel(
    x_ptr,        # pointer to input (B*C, N)
    out_ptr,      # pointer to output (B*C, N)
    scale_ptr,    # pointer to scale (C,) - scale is per-channel
    Bc,           # number of rows = B*C
    N,            # number of elements per row (spatial dims flattened)
    C,            # number of channels (to index scale)
    clamp_min,    # float
    clamp_max,    # float
    BLOCK: tl.constexpr
):
    """
    Each program handles one row (one (batch,channel) pair).
    It computes: y = softmax(clamp(x, clamp_min, clamp_max)) along the N dimension,
    then multiplies by scale[channel].
    """

    row = tl.program_id(0)  # which row (0 .. Bc-1)
    # If row >= Bc, exit early (grid may be larger due to rounding)
    if row >= Bc:
        return

    offs = tl.arange(0, BLOCK)
    row_start = row * N
    idx = row_start + offs
    mask = offs < N

    # Load and clamp in blocks
    # Use a safe "other" value for out-of-bounds loads
    other_val = clamp_min
    x = tl.load(x_ptr + idx, mask=mask, other=other_val)
    x = tl.where(x < clamp_min, clamp_min, x)
    x = tl.where(x > clamp_max, clamp_max, x)

    # compute max for numerical stability across the row
    m = tl.max(x, axis=0)  # scalar

    # compute exponentials and sum
    ex = tl.exp(x - m)
    s = tl.sum(ex, axis=0)  # scalar

    # compute normalized softmax and apply scale
    # determine channel index for this row: c = row % C
    c = row % C
    scale_val = tl.load(scale_ptr + c)
    out = ex / s * scale_val

    # store result
    tl.store(out_ptr + idx, out, mask=mask)


def triton_softmax_clamp_scale(x: torch.Tensor, clamp_min: float, clamp_max: float, scale: torch.Tensor):
    """
    x: tensor (B, C, N) or (B, C, D, H, W) already flattened to (B, C, N) before calling.
    scale: tensor of shape (C,) or broadcastable to (C,)
    Returns tensor of same shape as x, with softmax applied over last dim after clamp and scaled.
    """
    assert x.is_cuda and scale.is_cuda, "Tensors must be on CUDA."
    # x must be float32
    assert x.dtype == torch.float32 and scale.dtype == torch.float32

    B, C, N = x.shape
    # reshape to (B*C, N) contiguous
    x_flat = x.contiguous().view(B * C, N)
    out_flat = torch.empty_like(x_flat)

    # scale should be (C,)
    scale_flat = scale.contiguous().view(C)

    # grid and launch
    grid = lambda meta: ( (B * C + meta["BLOCK"] - 1) // meta["BLOCK"], )
    _softmax_clamp_scale_kernel[grid](
        x_flat, out_flat, scale_flat, B * C, N, C, float(clamp_min), float(clamp_max)
    )
    return out_flat.view(B, C, N)


class ModelNew(nn.Module):
    """
    Optimized Model:
      - Uses native nn.AvgPool3d and nn.ConvTranspose3d (kept as PyTorch ops).
      - Replaces clamp + softmax (over spatial dims) + multiplication by scale
        with a fused Triton kernel that does: clamp -> softmax -> scale per channel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.avg_pool = nn.AvgPool3d(pool_kernel_size)
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.register_buffer("_clamp_min", torch.tensor(float(clamp_min), dtype=torch.float32))
        self.register_buffer("_clamp_max", torch.tensor(float(clamp_max), dtype=torch.float32))
        # scale parameter per channel
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1, dtype=torch.float32))

    def forward(self, x):
        """
        x: (batch_size, in_channels, depth, height, width)
        returns: (batch_size, out_channels, depth, height, width)
        """
        x = self.avg_pool(x)
        x = self.conv_transpose(x)
        # At this point x is (B, C, D, H, W)
        b, c, d, h, w = x.shape
        # flatten spatial dims
        N = d * h * w
        x_flat = x.view(b, c, N)  # shape (B, C, N)
        # Prepare scale vector of shape (C,)
        scale_vec = self.scale.view(c).to(x.device)

        # Call Triton fused kernel: clamp -> softmax over dim=2 -> multiply by scale
        # Ensure contiguous and on CUDA
        if x_flat.is_cuda:
            out_flat = triton_softmax_clamp_scale(x_flat, float(self._clamp_min.item()), float(self._clamp_max.item()), scale_vec)
        else:
            # fallback to CPU implementation (pure PyTorch) to remain functional on CPU
            x_clamped = torch.clamp(x_flat, float(self._clamp_min.item()), float(self._clamp_max.item()))
            x_soft = torch.softmax(x_clamped, dim=2)
            out_flat = x_soft * scale_vec.view(1, c, 1)

        out = out_flat.view(b, c, d, h, w)
        return out