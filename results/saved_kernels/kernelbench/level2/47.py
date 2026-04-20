import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configurations for the elementwise fused activation kernel
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2),
]

@triton.autotune(
    configs=AUTOTUNE_CONFIGS,
    key=["n_elements"]
)
@triton.jit
def mish_tanh_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel that computes out = tanh( x * tanh( log(1 + exp(x)) ) )
    This fuses Mish activation followed by a tanh.
    """
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load input
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # softplus = log(1 + exp(x))
    exp_x = tl.exp(x)
    one_plus_exp = 1.0 + exp_x
    softplus = tl.log(one_plus_exp)

    # tanh(softplus) implemented via exp to avoid tl.tanh (not available)
    two_sp = softplus + softplus
    exp_two_sp = tl.exp(two_sp)
    tanh_sp = (exp_two_sp - 1.0) / (exp_two_sp + 1.0)

    # mish = x * tanh(softplus)
    mish = x * tanh_sp

    # tanh(mish) again via exp
    two_mish = mish + mish
    exp_two_m = tl.exp(two_mish)
    out = (exp_two_m - 1.0) / (exp_two_m + 1.0)

    # Store output
    tl.store(out_ptr + offs, out, mask=mask)


def triton_mish_tanh(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper to launch the Triton kernel for fused Mish + Tanh.
    Accepts a contiguous cuda float32 tensor and returns a new tensor.
    """
    assert x.is_cuda, "Input must be on CUDA."
    assert x.dtype == torch.float32, "Only float32 is supported by this kernel."

    x_contig = x.contiguous()
    out = torch.empty_like(x_contig)

    n_elements = x_contig.numel()
    # grid based on BLOCK_SIZE meta
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    mish_tanh_kernel[grid](x_contig, out, n_elements)
    return out


class ModelNew(nn.Module):
    """
    Optimized model: uses the standard nn.Conv3d for convolution (leveraging cuDNN),
    and a fused Triton kernel to compute Mish followed by Tanh in one pass.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D', H', W').
        """
        x = self.conv(x)
        # Apply fused Mish -> Tanh via Triton kernel for improved elementwise performance.
        if x.is_cuda and x.dtype == torch.float32:
            x = triton_mish_tanh(x)
        else:
            # Fallback to PyTorch implementation for CPU or other dtypes
            x = torch.nn.functional.mish(x)
            x = torch.tanh(x)
        return x


# Keep the input generation helpers compatible with the original interface
batch_size = 16
in_channels = 32
out_channels = 64
D, H, W = 32, 64, 64
kernel_size = 3

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]