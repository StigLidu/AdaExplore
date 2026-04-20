import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configs for the elementwise kernel
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 256},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 512},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2),
]

@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["n_elements"])
@triton.jit
def _clamp_divide_kernel(x_ptr, out_ptr, n_elements, min_value, divisor, BLOCK_SIZE: tl.constexpr):
    """
    Each program processes a contiguous block of BLOCK_SIZE elements.
    Performs: out = max(x, min_value) / divisor
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # clamp: if x < min_value -> min_value else x
    clamped = tl.where(x < min_value, min_value, x)
    res = clamped / divisor
    tl.store(out_ptr + offsets, res, mask=mask)

def triton_clamp_divide(x: torch.Tensor, min_value: float, divisor: float) -> torch.Tensor:
    """
    Wrapper to call the Triton kernel that clamps to min_value and divides by divisor.
    """
    assert x.is_cuda, "Input must be on CUDA"
    x = x.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    if n_elements == 0:
        return out

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    # Launch kernel; min_value and divisor are passed as scalar floats
    _clamp_divide_kernel[grid](x, out, n_elements, float(min_value), float(divisor))
    return out

class ModelNew(nn.Module):
    """
    Optimized model: uses native ConvTranspose3d for the heavy convolution,
    and a fused Triton kernel for clamp(min=...) followed by division.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.min_value = float(min_value)
        self.divisor = float(divisor)

    def forward(self, x):
        x = self.conv_transpose(x)
        # Fuse clamp(min=...) and division into a single Triton kernel for speed.
        x = triton_clamp_divide(x, self.min_value, self.divisor)
        return x

# Keep the helper functions similar to the original definitions for compatibility
batch_size = 16
in_channels = 64
out_channels = 128
depth, height, width = 24, 48, 48
kernel_size = 3
stride = 2
padding = 1
min_value = -1.0
divisor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, min_value, divisor]