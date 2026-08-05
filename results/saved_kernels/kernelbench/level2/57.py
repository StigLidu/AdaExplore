import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configurations for the 1D fused elementwise kernel
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 256},  num_warps=2, num_stages=2),
    triton.Config({"BLOCK": 512},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=3),
]

@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['n_elements'])
@triton.jit
def _fused_relu_hardswish_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    """
    Each program processes a contiguous block of up to BLOCK elements from the input.
    The kernel applies:
      rel = max(x, 0)
      out = rel * clamp((rel + 3) / 6, 0, 1)
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK
    offsets = block_start + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # ReLU: rel = max(vals, 0)
    rel = tl.where(vals > 0.0, vals, 0.0)

    # HardSwish on rel: rel * clamp((rel + 3)/6, 0, 1)
    tmp = (rel + 3.0) / 6.0
    tmp_clamped = tl.where(tmp < 0.0, 0.0, tl.where(tmp > 1.0, 1.0, tmp))

    out = rel * tmp_clamped

    tl.store(out_ptr + offsets, out, mask=mask)


def triton_fused_relu_hardswish(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper to launch the Triton kernel. Expects x on CUDA.
    Preserves the input shape.
    """
    assert x.is_cuda, "Input must be on CUDA."
    # Make contiguous and flatten for 1D kernel
    original_shape = x.shape
    x_flat = x.contiguous().view(-1)
    n_elements = x_flat.numel()

    out_flat = torch.empty_like(x_flat)

    grid = lambda meta: ((n_elements + meta["BLOCK"] - 1) // meta["BLOCK"],)

    # Launch kernel
    _fused_relu_hardswish_kernel[grid](x_flat, out_flat, n_elements)
    return out_flat.view(original_shape)


class ModelNew(nn.Module):
    """
    Optimized model that uses the original Conv2d for convolution
    and a Triton kernel to fuse ReLU + HardSwish elementwise activation.
    This preserves the original behavior while offloading the activation
    to a custom GPU kernel for improved performance.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = triton_fused_relu_hardswish(x)
        return x


# Keep helper functions similar to the original module for compatibility
batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]