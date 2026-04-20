import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotuning configs for the Triton kernel
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=2),
]


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['n_rows', 'HW'])
@triton.jit
def _fused_bias_clamp_scale_kernel(
    x_ptr,          # pointer to input (conv_transpose output), flattened as (N*C, H*W)
    bias_ptr,       # pointer to bias of shape (C,)
    out_ptr,        # pointer to output
    n_rows,         # N * C
    HW,             # H * W
    C,              # channels
    hi,             # upper clamp bound (precomputed min(1.0, 1.0/scaling))
    BLOCK: tl.constexpr,
):
    # Each program handles one contiguous spatial tile of a single (n, c) row.
    pid = tl.program_id(0)

    # how many tiles per row
    tiles_per_row = (HW + BLOCK - 1) // BLOCK

    # decode which row and which tile within the row this program handles
    row = pid // tiles_per_row
    tile = pid % tiles_per_row

    # start offset within the row
    block_start = tile * BLOCK
    offs = block_start + tl.arange(0, BLOCK)
    mask = offs < HW

    # compute flattened linear offsets into the original flattened tensor (row-major over H*W)
    linear_offs = row * HW + offs

    # load input values for this tile
    x_vals = tl.load(x_ptr + linear_offs, mask=mask, other=0.0)

    # load the bias for this (n,c) row once and broadcast to lanes
    ch = row % C
    # load bias scalar and broadcast to lanes
    bias_scalar = tl.load(bias_ptr + ch)
    bias_vals = bias_scalar + tl.zeros((BLOCK,), dtype=tl.float32)

    # fused operations simplified:
    # out = clamp(x + bias, 0, hi)
    out = x_vals + bias_vals
    out = tl.where(out < 0.0, 0.0, out)
    out = tl.where(out > hi, hi, out)

    # store results
    tl.store(out_ptr + linear_offs, out, mask=mask)


def triton_fused_bias_clamp_scale(x: torch.Tensor, bias: torch.Tensor, scaling: float):
    """
    Wrapper to call the Triton fused kernel.
    Expects:
      x: Tensor (N, C, H, W) on CUDA, contiguous
      bias: Tensor (C, 1, 1) or (C,) on CUDA
    Returns:
      out: Tensor same shape as x
    """
    assert x.is_cuda and bias.is_cuda, "Inputs must be on CUDA."
    x = x.contiguous()
    N, C, H, W = x.shape

    # view tensor as (N*C, H*W) so each program can operate on a tile of a single row
    n_rows = N * C
    HW = H * W

    # prepare bias as 1D contiguous tensor of length C
    bias_flat = bias.contiguous().view(-1)

    # precompute reduced upper clamp bound hi = min(1.0, 1.0/scaling)
    hi = float(min(1.0, 1.0 / float(scaling)))

    out = torch.empty_like(x)

    # grid calculation: one program per tile; total programs = n_rows * tiles_per_row
    grid = lambda meta: (n_rows * ((HW + meta['BLOCK'] - 1) // meta['BLOCK']),)

    # launch kernel
    _fused_bias_clamp_scale_kernel[grid](
        x,                      # x_ptr (tensor)
        bias_flat,              # bias_ptr (tensor)
        out,                    # out_ptr (tensor)
        n_rows,                 # N * C
        HW,                     # H * W
        C,                      # channels
        hi                      # hi scalar
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model: keep PyTorch ConvTranspose2d, fuse bias add + clamp + scale + clamp + divide
    into a single Triton kernel for improved elementwise performance.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                                 stride=stride, padding=padding, output_padding=output_padding)
        # keep bias as same shape as original for compatibility; Triton wrapper will flatten it
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv_transpose(x)
        # Use Triton fused kernel for bias add + clamp + scale + clamp + divide
        return triton_fused_bias_clamp_scale(x, self.bias, self.scaling_factor)


# Inputs and init inputs to match original interface
batch_size = 128
in_channels  = 64
out_channels = 64
height = width = 128
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0


def get_inputs():
    # Note: Triton kernel requires CUDA tensors at runtime; the test harness may move tensors to GPU.
    return [torch.rand(batch_size, in_channels, height, width).cuda()]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor]