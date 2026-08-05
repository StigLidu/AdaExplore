import torch
import torch.nn as nn
import triton
import triton.language as tl

# Fused Triton kernel: in-place on conv output
# Performs: y = x + add_val; y = min(y, 0.0); y = gelu_approx(y); y = y * mul_val
# Processes BLOCK * V elements per program (V elements per lane).
@triton.jit
def fused_elemwise_inplace_kernel(
    x_ptr,            # pointer to input/output
    n_elements,       # total number of elements
    add_val,          # scalar add value (fp32)
    mul_val,          # scalar multiply value (fp32)
    BLOCK: tl.constexpr,  # number of program lanes
    V: tl.constexpr       # vectorization (elements per lane)
):
    pid = tl.program_id(0)
    # each program handles BLOCK * V elements
    start = pid * BLOCK * V
    # build linear offsets directly (BLOCK and V are constexpr so BLOCK * V is compile-time)
    offs = start + tl.arange(0, BLOCK * V)                 # shape [BLOCK * V]
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # add
    y = x + add_val
    # min with 0.0 (keep negatives)
    # After this, positives are zeroed, so GELU(0)=0 and we can avoid extra work conceptually.
    neg_mask = y < 0.0
    y = tl.where(neg_mask, y, 0.0)

    # GELU approximation: x * sigmoid(1.702 * x)
    # Compute using the clamped values; positives are already zero so result will be zero for them.
    z = 1.702 * y
    s = 1.0 / (1.0 + tl.exp(-z))
    y = y * s

    # final multiply
    y = y * mul_val

    tl.store(x_ptr + offs, y, mask=mask)


def triton_fused_inplace(x: torch.Tensor, add_val: float, mul_val: float):
    """
    Wrapper for the fused Triton kernel.
    - Operates in-place on x (the conv_transpose output).
    - x is flattened; kernel performs masked loads/stores and writes back to x.
    """
    assert x.is_cuda, "Input must be on CUDA"
    assert x.dtype == torch.float32, "Only fp32 supported"
    x_flat = x.contiguous().view(-1)
    n_elements = x_flat.numel()

    # Tunable parameters for Ampere (A6000)
    # Choose BLOCK (threads per program) and V (vectorization per thread).
    # BLOCK is constexpr; V is constexpr. Each program handles BLOCK * V elements.
    # Use a smaller BLOCK for better occupancy on Ampere; V keeps loads vectorized.
    BLOCK = 256   # number of program lanes
    V = 4         # elements per lane -> each program handles 1024 elements

    grid = lambda meta: ((n_elements + meta["BLOCK"] * V - 1) // (meta["BLOCK"] * V),)
    fused_elemwise_inplace_kernel[grid](x_flat, n_elements, float(add_val), float(mul_val), BLOCK=BLOCK, V=V)
    return x_flat.view_as(x)


class ModelNew(nn.Module):
    """
    Optimized model:
      - Uses the original nn.ConvTranspose2d for the heavy convolution transpose.
      - Fuses add -> min(.,0) -> GELU -> multiply into a single Triton kernel,
        operating in-place on the conv output to avoid intermediate allocations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = float(add_value)
        self.multiply_value = float(multiply_value)

    def forward(self, x):
        # conv_transpose uses efficient PyTorch CUDA implementation
        x = self.conv_transpose(x)
        # ensure contiguous so Triton can safely flatten and write in-place
        x = x.contiguous()
        # fused in-place elementwise operations
        x = triton_fused_inplace(x, self.add_value, self.multiply_value)
        return x


# Keep input generation helpers compatible with expected harness
batch_size = 128
in_channels = 64
out_channels = 128
height, width = 64, 64
kernel_size = 4
stride = 2
add_value = 0.5
multiply_value = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda().to(torch.float32)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, add_value, multiply_value]