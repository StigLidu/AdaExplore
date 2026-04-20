import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configurations for the Triton kernel
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 2048}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 4096}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK": 8192}, num_warps=8, num_stages=3),
]

@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['n_elements'])
@triton.jit
def postprocess_kernel(
    inp_ptr,       # pointer to conv output (float32)
    bias_ptr,      # pointer to bias (C,) flattened
    out_ptr,       # pointer to destination buffer
    const_val,     # scalar float32
    scale_val,     # scalar float32
    n_elements,    # total number of elements in inp/out
    C, H, W,       # ints: channel, height, width
    BLOCK: tl.constexpr
):
    # compute global element indices handled by this program
    block_start = tl.program_id(0) * BLOCK
    offsets = block_start + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    # Load input values (conv output)
    x = tl.load(inp_ptr + offsets, mask=mask, other=0.0)

    # compute channel index for each flattened element:
    # layout is assumed N, C, H, W with contiguous memory, so:
    # within_image = offsets % (C*H*W)
    # c_idx = within_image // (H*W)
    CHW = C * H * W
    within = offsets % CHW
    c_idx = within // (H * W)

    # Load bias per channel (broadcasted over spatial & batch)
    bias_vals = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

    # elementwise min with constant: result = min(x, const_val)
    # use tl.where to avoid relying on tl.minimum
    const_broadcast = const_val
    y = tl.where(x <= const_broadcast, x, const_broadcast)

    # add bias and scale
    y = (y + bias_vals) * scale_val

    # store result
    tl.store(out_ptr + offsets, y, mask=mask)


def triton_postprocess(inp: torch.Tensor, bias: torch.Tensor, const_val: float, scale_val: float):
    """
    Applies elementwise: out = min(inp, const_val) + bias_broadcasted; then scales by scale_val.
    inp: Tensor of shape (N, C, H, W), float32, CUDA
    bias: Tensor of shape (C, 1, 1) or (C,), float32, CUDA
    """
    assert inp.is_cuda and bias.is_cuda, "Tensors must be on CUDA"
    assert inp.dtype == torch.float32 and bias.dtype == torch.float32, "Only float32 supported"

    x = inp.contiguous()
    N, C, H, W = x.shape
    n_elements = x.numel()

    # Flatten bias to shape (C,)
    bias_flat = bias.contiguous().view(-1)

    # Prepare output tensor
    out = torch.empty_like(x)

    # grid based on BLOCK meta
    grid = lambda meta: ((n_elements + meta["BLOCK"] - 1) // meta["BLOCK"],)

    # launch kernel
    postprocess_kernel[grid](
        x,
        bias_flat,
        out,
        float(const_val),
        float(scale_val),
        n_elements,
        C, H, W
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that uses Triton to fuse the elementwise post-processing:
      out = (min(conv(x), constant_value) + bias) * scaling_factor
    The convolution itself uses the standard PyTorch Conv2d implementation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # keep the external bias (separate from conv's own bias)
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        # expect x on CUDA for Triton kernel
        x = self.conv(x)
        # Ensure tensors are on the same device and dtype
        bias = self.bias
        if not x.is_cuda:
            # fallback to CPU path if input is CPU: perform operations in PyTorch
            x = torch.min(x, torch.tensor(self.constant_value, dtype=x.dtype, device=x.device))
            x = x + bias
            x = x * self.scaling_factor
            return x
        # Triton fused postprocessing (runs on CUDA)
        return triton_postprocess(x, bias, self.constant_value, self.scaling_factor)


# Utilities to mirror the original module's helper functions
batch_size = 128
in_channels = 64
out_channels = 128
height = width = 128
kernel_size = 3
constant_value = 0.5
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor]