import torch
import torch.nn as nn
import triton
import triton.language as tl

# Tuned block sizes to try; choose a larger block to increase memory bandwidth utilization
# while still being a multiple of 32 (warp size). We avoid autotune complexity and pick a
# single well-performing BLOCK for A6000.
_BLOCK = 4096  # must be multiple of 32


@triton.jit
def _fused_pointwise_kernel(
    x_ptr,        # pointer to input tensor (fp32)
    out_ptr,      # pointer to output tensor (fp32) - can alias x_ptr for in-place
    add_val,      # float scalar: value to add
    mul_val,      # float scalar: value to multiply
    n_elements,   # total number of elements (int)
    BLOCK: tl.constexpr,  # compile-time block size
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements

    # Load values (masked)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # add
    x = x + add_val

    # min(x, 0.0) -> keep negative values, clamp positives to 0
    x = tl.where(x < 0.0, x, 0.0)

    # GELU approximation: x * sigmoid(1.702 * x)
    # sigmoid(z) = 1 / (1 + exp(-z))
    sig = 1.0 / (1.0 + tl.exp(-1.702 * x))
    x = x * sig

    # multiply
    x = x * mul_val

    # Store result (masked)
    tl.store(out_ptr + offs, x, mask=mask)


def fused_pointwise_inplace(x: torch.Tensor, add_value: float, multiply_value: float):
    """
    Fuse the sequence of elementwise ops (add, min(x,0), GELU approximation, multiply)
    into a single Triton kernel and run it in-place on the provided tensor to avoid
    an extra allocation/copy. The tensor is flattened and processed as contiguous memory.
    """
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()
    out = x  # in-place

    n_elements = x.numel()
    if n_elements == 0:
        return out

    BLOCK = _BLOCK
    num_blocks = (n_elements + BLOCK - 1) // BLOCK
    grid = (num_blocks,)

    # Launch kernel (BLOCK is a constexpr argument)
    _fused_pointwise_kernel[grid](
        x, out,
        float(add_value),
        float(multiply_value),
        n_elements,
        BLOCK=BLOCK,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model:
      - Uses PyTorch's nn.ConvTranspose2d for the transposed convolution (leveraging cuDNN/cuBLAS).
      - Fuses the subsequent elementwise operations (add, min with 0, GELU, multiply)
        into a single Triton kernel which operates in-place to avoid extra allocations and copies.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        # store scalars as Python floats for passing to Triton
        self.add_value = float(add_value)
        self.multiply_value = float(multiply_value)

    def forward(self, x):
        # conv transpose using highly-optimized PyTorch implementation
        x = self.conv_transpose(x)
        # fused elementwise operations via Triton (in-place)
        x = fused_pointwise_inplace(x, self.add_value, self.multiply_value)
        return x


# Model parameters and helper functions for compatibility
batch_size = 128
in_channels = 64
out_channels = 128
height, width = 64, 64
kernel_size = 4
stride = 2
add_value = 0.5
multiply_value = 2.0

def get_inputs():
    # Return a CUDA tensor for inference
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, add_value, multiply_value]