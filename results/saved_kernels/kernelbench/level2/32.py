import torch
import torch.nn as nn
import triton
import triton.language as tl

# More aggressive autotune configurations to explore larger spatial tiles and channel blocks.
AUTOTUNE_CONFIGS = [
    triton.Config({"TILE_HW": 128,  "CHANNEL_BLOCK": 8},  num_warps=2, num_stages=2),
    triton.Config({"TILE_HW": 256,  "CHANNEL_BLOCK": 16}, num_warps=4, num_stages=2),
    triton.Config({"TILE_HW": 512,  "CHANNEL_BLOCK": 16}, num_warps=4, num_stages=2),
    triton.Config({"TILE_HW": 1024, "CHANNEL_BLOCK": 32}, num_warps=8, num_stages=3),
    triton.Config({"TILE_HW": 2048, "CHANNEL_BLOCK": 32}, num_warps=8, num_stages=3),
]

@triton.autotune(
    configs=AUTOTUNE_CONFIGS,
    key=['N', 'C', 'H', 'W'],
)
@triton.jit
def _min_kernel(
    x_ptr,          # pointer to input tensor (N, C, H, W) flattened
    out_ptr,        # pointer to output tensor (N, 1, H, W) flattened
    N, C, H, W,     # dimensions
    TILE_HW: tl.constexpr,      # number of spatial elements per program (constexpr)
    CHANNEL_BLOCK: tl.constexpr,# number of channels to process per inner loop (constexpr)
):
    """
    Each program reduces over CHANNELS for TILE_HW spatial locations for a single batch element.
    This kernel assumes the input has already been scaled (i.e., any scalar multiplication
    was folded into the convolution weights) so it only computes minimum across channels.
    """
    pid = tl.program_id(0)  # one program per (batch, spatial_tile)
    HW = H * W
    tiles_per_n = (HW + TILE_HW - 1) // TILE_HW

    # compute batch index and tile index from pid
    b = pid // tiles_per_n
    tile_id = pid - b * tiles_per_n

    # spatial offsets for this tile (contiguous across spatial dim)
    base_off = tile_id * TILE_HW
    offs = base_off + tl.arange(0, TILE_HW)          # shape (TILE_HW,)
    mask_spatial = offs < HW                         # spatial tail mask

    # initialize accumulator to large positive value for min
    init_val = 1e30
    acc = tl.full((TILE_HW,), init_val, dtype=tl.float32)

    # iterate channels in blocks of CHANNEL_BLOCK
    c_start = 0
    while c_start < C:
        for ci in range(CHANNEL_BLOCK):
            c = c_start + ci
            mask_c = c < C
            # flattened index for contiguous load of spatial tile for channel c:
            # ((b * C + c) * H * W) + offs
            idx = ((b * C + c) * HW) + offs
            mask = mask_spatial & mask_c
            vals = tl.load(x_ptr + idx, mask=mask, other=init_val)
            acc = tl.minimum(acc, vals)
        c_start += CHANNEL_BLOCK

    # store the TILE_HW results to the output (with spatial mask for the tail)
    out_base = b * HW  # output has shape (N, 1, H, W) flattened so stride is HW per batch
    out_idx = out_base + offs
    tl.store(out_ptr + out_idx, acc, mask=mask_spatial)


def triton_min(x: torch.Tensor):
    """
    Wrapper that runs the Triton kernel to compute min over channels.
    Input:
      x: Tensor of shape (N, C, H, W), contiguous, CUDA, float32
    Output:
      Tensor of shape (N, 1, H, W), same device/dtype as x
    """
    assert x.is_cuda and x.dtype == torch.float32, "Input must be CUDA float32 tensor"
    x = x.contiguous()

    N, C, H, W = x.shape
    out = torch.empty((N, 1, H, W), device=x.device, dtype=x.dtype)

    x_flat = x.view(-1)
    out_flat = out.view(-1)

    HW = H * W

    # grid: one program per (batch, spatial_tile)
    grid = lambda meta: (N * ((HW + meta['TILE_HW'] - 1) // meta['TILE_HW']),)

    # launch kernel (autotune will pick TILE_HW and CHANNEL_BLOCK)
    _min_kernel[grid](
        x_flat,
        out_flat,
        N, C, H, W,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model:
      - Use standard nn.Conv2d (highly optimized) for the convolution.
      - Fold the scalar scale_factor into conv weights/bias at initialization so
        we avoid an extra elementwise multiplication after convolution.
      - Use a Triton kernel to compute the minimum across the channel dimension
        in a fused, high-throughput manner.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # Fold the scale factor into conv parameters to remove a separate multiply
        if scale_factor != 1.0:
            with torch.no_grad():
                self.conv.weight.mul_(float(scale_factor))
                if self.conv.bias is not None:
                    self.conv.bias.mul_(float(scale_factor))

    def forward(self, x):
        x = self.conv(x)
        return triton_min(x)


# Keep helper functions consistent with the original interface
batch_size = 64
in_channels = 64
out_channels = 128
height = width = 256
kernel_size = 3
scale_factor = 2.0

def get_inputs():
    # Triton kernel expects CUDA tensors
    return [torch.rand(batch_size, in_channels, height, width, device='cuda', dtype=torch.float32)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]