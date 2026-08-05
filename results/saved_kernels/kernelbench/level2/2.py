import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configs for the Triton kernel
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 256},  num_warps=2, num_stages=2),
    triton.Config({"BLOCK": 512},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=3),
]

@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['n_elements'])
@triton.jit
def fused_postproc_kernel(
    x_ptr,           # pointer to input tensor data (conv_transpose output)
    bias_ptr,        # pointer to bias data (C,)
    out_ptr,         # pointer to output tensor data
    n_elements,      # total number of elements in the tensor (B*C*H*W)
    C,               # channels (out_channels)
    H,               # height
    W,               # width
    scaling,         # scaling factor (float32)
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    start = pid * BLOCK
    offs = start + tl.arange(0, BLOCK)
    mask = offs < n_elements

    # Load input values (other=0.0 for out-of-bounds)
    vals = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # Compute channel index for each linear offset:
    # Given linear index offs in [0, B*C*H*W),
    # hw = H*W
    # channel = (offs // hw) % C
    hw = H * W
    # Use arithmetic avoiding modulus operator for better compatibility:
    div_hw = offs // hw                       # in [0, B*C)
    div_hwC = div_hw // C                     # batch index
    channel = div_hw - div_hwC * C            # channel index in [0, C)

    # Load bias per-channel (broadcast)
    b = tl.load(bias_ptr + channel, mask=mask, other=0.0)

    # Fused elementwise operations:
    # v = vals + bias
    # v = clamp(v, 0, 1)
    # v = v * scaling
    # v = clamp(v, 0, 1)
    # v = v / scaling
    v = vals + b

    # first clamp to [0,1]
    v = tl.where(v < 0.0, 0.0, v)
    v = tl.where(v > 1.0, 1.0, v)

    # scale
    v = v * scaling

    # second clamp to [0,1]
    v = tl.where(v < 0.0, 0.0, v)
    v = tl.where(v > 1.0, 1.0, v)

    # divide by scaling
    v = v / scaling

    # store result
    tl.store(out_ptr + offs, v, mask=mask)


def triton_fused_postproc(x: torch.Tensor, bias: torch.Tensor, scaling: float):
    """
    Wraps the Triton fused kernel.
    x: [B, C, H, W] tensor (conv_transpose output), cuda, contiguous or will be made contiguous
    bias: [C,1,1] or [C] tensor (Parameter), cuda
    scaling: float scalar
    """
    assert x.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    # Ensure bias is contiguous 1D of length C
    b = bias.contiguous().view(-1)

    B, C, H, W = x.shape
    n_elements = x.numel()

    out = torch.empty_like(x)

    # grid based on autotuning meta BLOCK
    grid = lambda meta: ((n_elements + meta['BLOCK'] - 1) // meta['BLOCK'],)

    fused_postproc_kernel[grid](
        x, b, out,
        n_elements,
        C, H, W,
        float(scaling)
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model: keeps ConvTranspose2d in PyTorch, but fuses bias add,
    clamps, scaling, and divide into a single Triton kernel for one-pass elementwise processing.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        # maintain separate bias as in original model
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        # conv transpose in PyTorch
        x = self.conv_transpose(x)
        # fused elementwise postprocessing on GPU
        x = triton_fused_postproc(x, self.bias, self.scaling_factor)
        return x


# Reproduce the original input generation utilities
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
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor]