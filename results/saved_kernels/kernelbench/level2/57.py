import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configs for the elementwise fused activation kernel
# Tuned for A6000: keep moderate BLOCK sizes to balance occupancy and register/shared pressure.
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 512}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=2),
]

@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['n_elements'])
@triton.jit
def _fused_relu_hardswish_kernel(
    x_ptr,         # input pointer
    out_ptr,       # output pointer
    n_elements,    # number of elements (flattened)
    BLOCK: tl.constexpr
):
    # each program processes a contiguous block of size BLOCK
    block_start = tl.program_id(0) * BLOCK
    offsets = block_start + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    # load inputs with masking
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # ReLU
    x_relu = tl.maximum(x, 0.0)

    # HardSwish: x * clamp((x + 3) / 6, 0, 1)
    a = tl.minimum(tl.maximum((x_relu + 3.0) / 6.0, 0.0), 1.0)
    out = x_relu * a

    # store outputs
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_fused_relu_hardswish(x: torch.Tensor) -> torch.Tensor:
    """
    Apply fused ReLU + HardSwish (x * clamp((x + 3) / 6, 0, 1)) using a Triton kernel.
    Works on CUDA tensors only. Preserves shape.
    """
    if not x.is_cuda:
        # fallback to pure PyTorch for CPU
        x = torch.relu(x)
        return x * torch.clamp((x + 3.0) / 6.0, 0.0, 1.0)

    # operate on flattened contiguous tensor for coalesced memory accesses
    if x.is_contiguous():
        x_flat = x.view(-1)
    else:
        x_flat = x.contiguous().view(-1)

    n_elements = x_flat.numel()

    # allocate separate output buffer to avoid in-place aliasing issues
    out_flat = x_flat.new_empty(n_elements)

    # grid based on selected BLOCK size from autotune metadata
    grid = lambda meta: ((n_elements + meta['BLOCK'] - 1) // meta['BLOCK'],)

    # launch kernel (read from x_flat, write to out_flat)
    _fused_relu_hardswish_kernel[grid](x_flat, out_flat, n_elements)

    # reshape back to original shape
    return out_flat.view_as(x)


class ModelNew(nn.Module):
    """
    Optimized model:
      - Uses the PyTorch Conv2d for the convolution (leveraging cuDNN/CUDA)
      - Fuses ReLU and HardSwish into a single Triton kernel for the activation
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        # keep the same convolution as original for correctness and performance
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        # fuse ReLU + HardSwish with a Triton kernel for elementwise speedup
        x = triton_fused_relu_hardswish(x)
        return x


# Maintain the original helper functions for compatibility
batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]