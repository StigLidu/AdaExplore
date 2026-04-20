import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotuning configs for the reduction kernel
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=3),
]

@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['n_rows', 'per_row'])
@triton.jit
def _mul_and_global_mean_kernel(
    x_ptr,            # pointer to input tensor (N*C*H*W elements)
    out_ptr,          # pointer to output tensor (N*C elements)
    multiplier,       # scalar multiplier (float32)
    per_row,          # number of spatial elements per (n,c) = H*W
    n_rows,           # number of rows = N * C
    BLOCK_SIZE: tl.constexpr
):
    """
    Each program reduces one (n, c) row: it computes mean over per_row elements.
    The input is expected to be contiguous in memory in NCHW layout, so each row
    is a contiguous block of size `per_row`.
    """
    row = tl.program_id(0)
    # Boundary check
    if row >= n_rows:
        return

    # Accumulator for the sum (scalar)
    acc = 0.0

    # Base pointer for this row
    base = row * per_row

    # Loop over the row in chunks of BLOCK_SIZE
    start = 0
    while start < per_row:
        offs = tl.arange(0, BLOCK_SIZE)
        idx = base + start + offs
        mask = offs < (per_row - start)
        vals = tl.load(x_ptr + idx, mask=mask, other=0.0)
        # multiply and accumulate
        acc = acc + tl.sum(vals * multiplier, 0)
        start += BLOCK_SIZE

    # Compute mean and store result
    mean = acc / per_row
    tl.store(out_ptr + row, mean)


def triton_mul_and_global_mean(x: torch.Tensor, multiplier: float) -> torch.Tensor:
    """
    Wrapper to launch the Triton kernel that multiplies the input by a scalar and
    computes a global spatial mean per (N, C), producing output shaped (N, C, 1, 1).
    """
    assert x.is_cuda, "Input must be on CUDA"
    assert x.dtype == torch.float32, "This kernel assumes float32 tensors"

    x = x.contiguous()
    N, C, H, W = x.shape
    per_row = H * W
    n_rows = N * C

    # Prepare output (N*C,) and reshaped to (N,C,1,1) later
    out_flat = torch.empty(n_rows, device=x.device, dtype=x.dtype)

    # Create grid
    grid = lambda meta: ( (n_rows + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'], )

    # Launch kernel
    _mul_and_global_mean_kernel[grid](
        x,                     # x_ptr (Triton accepts torch.Tensor directly)
        out_flat,              # out_ptr
        float(multiplier),     # multiplier
        per_row,               # per_row
        n_rows,                # n_rows
    )

    # Reshape to (N, C, 1, 1)
    out = out_flat.view(N, C, 1, 1)
    return out


class ModelNew(nn.Module):
    """
    Optimized model:
      - Uses the PyTorch nn.ConvTranspose2d for the transposed convolution (to keep correctness & use optimized library code)
      - Replaces the subsequent elementwise multiply and two global means with a single Triton kernel
        that multiplies and computes the spatial global mean per (N, C) without materializing extra tensors.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        # store multiplier as float
        self.multiplier = float(multiplier)

    def forward(self, x):
        # 1) Transposed convolution (use PyTorch's optimized kernel)
        x = self.conv_transpose(x)
        # 2) Triton kernel: multiply by scalar and compute global mean over spatial dims
        #    The original model applied mean twice; the second mean over a 1x1 spatial dimension
        #    is redundant, so one global mean is sufficient and yields the same result.
        out = triton_mul_and_global_mean(x, self.multiplier)
        return out