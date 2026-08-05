import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configs for different block sizes/warps
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 256},  num_warps=2, num_stages=2),
    triton.Config({"BLOCK": 512},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=3),
]

@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['n_elements'])
@triton.jit
def fused_mul_lrelu_gelu_kernel(
    x_ptr,            # input pointer (fp32)
    mult_ptr,         # multiplier pointer (fp32) length = C
    out_ptr,          # output pointer (fp32)
    n_elements,       # total number of elements (int)
    HW,               # H * W (int)
    C,                # number of channels (int)
    negative_slope,   # leaky relu negative slope (float)
    inv_sqrt2,        # 1/sqrt(2) constant for erf-based GELU (float)
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK
    offs = block_start + tl.arange(0, BLOCK)
    mask = offs < n_elements

    # Load inputs
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # Compute channel indices per element: c = (index // (H*W)) % C
    # Note: integer ops are supported on tl.arange results
    ch = (offs // HW) % C
    mult = tl.load(mult_ptr + ch, mask=mask, other=1.0)

    # Multiply
    y = x * mult

    # LeakyReLU: y = y if y>0 else y * negative_slope
    y = tl.where(y > 0.0, y, y * negative_slope)

    # GELU via erf: 0.5 * x * (1 + erf(x / sqrt(2)))
    g = 0.5 * y * (1.0 + tl.erf(y * inv_sqrt2))

    # Store
    tl.store(out_ptr + offs, g, mask=mask)


def fused_mul_lrelu_gelu(x: torch.Tensor, multiplier: torch.Tensor, negative_slope: float = 0.01):
    """
    x: tensor of shape (N, C, H, W), contiguous, on CUDA
    multiplier: tensor of shape (C, ) or (C,1,1), contiguous, on CUDA
    """
    assert x.is_cuda and multiplier.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    # Flatten multiplier to (C,)
    mult = multiplier.contiguous().view(-1)

    out = torch.empty_like(x)
    n_elements = x.numel()
    N, C, H, W = x.shape
    HW = H * W
    inv_sqrt2 = 1.0 / (2.0 ** 0.5)  # 1 / sqrt(2)

    # Grid
    grid = lambda meta: ((n_elements + meta["BLOCK"] - 1) // meta["BLOCK"],)

    # Launch
    fused_mul_lrelu_gelu_kernel[grid](
        x, mult, out,
        n_elements,
        HW,
        C,
        float(negative_slope),
        float(inv_sqrt2),
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized Model:
      - Uses PyTorch Conv2d for convolution.
      - Fuses multiply by learnable multiplier, LeakyReLU, and GELU into a single Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # multiplier stored in same shape as original (out_channels, 1, 1)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        # Keep attribute for negative_slope to mirror original LeakyReLU default
        self.negative_slope = 0.01

    def forward(self, x):
        x = self.conv(x)
        # fused elementwise operations on GPU
        return fused_mul_lrelu_gelu(x, self.multiplier, negative_slope=self.negative_slope)


# Preserve the original input helpers for compatibility
batch_size = 64
in_channels = 64
out_channels = 64
height, width = 256, 256
kernel_size = 3
multiplier_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, multiplier_shape]