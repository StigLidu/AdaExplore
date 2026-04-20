import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Autotune configs for the activation kernel
# Increase per-program work and vectorization to favor fewer launches with wider memory ops.
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 1024, "V": 4}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_SIZE": 512, "V": 8}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_SIZE": 256, "V": 8}, num_warps=4, num_stages=2),
]

@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['n_elements'])
@triton.jit
def _hswish_relu_kernel(
    x_ptr,             # pointer to input tensor
    out_ptr,           # pointer to output tensor
    n_elements,        # total number of elements
    BLOCK_SIZE: tl.constexpr,
    V: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * (BLOCK_SIZE * V)
    # Create a single contiguous range for this program: BLOCK_SIZE * V elements
    rng = tl.arange(0, BLOCK_SIZE * V)
    offs = base + rng
    mask = offs < n_elements

    # Single vectorized load for all work-per-program
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # Compute hardswish: x * clamp(x + 3, 0, 6) / 6
    tmp = x + 3.0
    # clamp tmp to [0, 6]
    tmp = tl.where(tmp <= 0.0, 0.0, tmp)
    tmp = tl.where(tmp >= 6.0, 6.0, tmp)
    y = x * (tmp / 6.0)

    # Apply ReLU: max(0, y)
    y = tl.where(y > 0.0, y, 0.0)

    # Single vectorized store
    tl.store(out_ptr + offs, y, mask=mask)

def triton_hardswish_relu(x: torch.Tensor, in_place: bool = False) -> torch.Tensor:
    """
    Applies fused HardSwish followed by ReLU using a Triton kernel.

    Args:
        x: input CUDA float32 tensor.
        in_place: if True, write results into `x` (requires x to be contiguous).
                  If False, a new output tensor is allocated.

    Returns:
        Tensor with the activation applied.
    """
    assert x.is_cuda, "Input must be on CUDA"
    assert x.dtype == torch.float32, "This kernel expects float32 tensors"

    if in_place:
        # For true in-place operation avoid creating a contiguous copy here.
        # Require caller to provide a contiguous tensor when requesting in-place updates.
        assert x.is_contiguous(), "in_place requires contiguous input tensor"
        out = x
    else:
        # Ensure contiguous input for efficient, wider vectorized loads and allocate output once.
        x = x.contiguous()
        out = torch.empty_like(x)

    n = x.numel()
    # grid function for autotuner: each program handles BLOCK_SIZE * V elements
    grid = lambda meta: ((n + (meta['BLOCK_SIZE'] * meta['V']) - 1) // (meta['BLOCK_SIZE'] * meta['V']),)

    _hswish_relu_kernel[grid](x, out, n)
    return out

class ModelNew(nn.Module):
    """
    Optimized Model: uses PyTorch Conv2d for convolution and a Triton kernel
    to fuse HardSwish + ReLU activation for speed.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        # Keep a standard Conv2d module so weights/biases are trainable and compatible
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        # Perform convolution using the existing PyTorch operator (keeps correctness & autograd)
        x = self.conv(x)
        # Fuse HardSwish + ReLU using a custom Triton kernel for elementwise speedup
        x = triton_hardswish_relu(x)
        return x