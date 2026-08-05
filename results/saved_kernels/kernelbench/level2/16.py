import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configs for different block sizes
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 128},  num_warps=1, num_stages=2),
    triton.Config({"BLOCK_SIZE": 256},  num_warps=2, num_stages=2),
    triton.Config({"BLOCK_SIZE": 512},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2),
]

@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['n_elements'])
@triton.jit
def _fused_mish_add_clamp_scale_kernel(
    x_ptr,        # input pointer
    out_ptr,      # output pointer
    n_elements,   # total number of elements
    add_value,    # scalar to add
    min_val,      # clamp min
    max_val,      # clamp max
    scale,        # scale factor
    BLOCK_SIZE: tl.constexpr
):
    """
    Each program handles a contiguous block of BLOCK_SIZE elements.
    Performs: out = clamp(mish(x) + add_value, min_val, max_val) * scale
    where mish(x) = x * tanh(softplus(x)) and softplus(x) = log(1 + exp(x))
    """
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # softplus: log(1 + exp(x))
    # Use tl.exp and tl.log
    # For stability, compute softplus in a numerically stable way:
    # softplus(x) = where(x > 20, x, log(1 + exp(x))) to avoid overflow
    # 20 is a safe threshold for float32
    large_mask = x > 20.0
    exp_x = tl.exp(x)
    softplus = tl.where(large_mask, x, tl.log(1.0 + exp_x))

    # tanh(softplus) computed via tanh identity: (e^{2s}-1)/(e^{2s}+1)
    two_s = 2.0 * softplus
    e2s = tl.exp(two_s)
    tanh_s = (e2s - 1.0) / (e2s + 1.0)

    mish = x * tanh_s

    y = mish + add_value

    # clamp (Hardtanh): clamp y between min_val and max_val
    y = tl.where(y < min_val, min_val, y)
    y = tl.where(y > max_val, max_val, y)

    out = y * scale

    tl.store(out_ptr + offsets, out, mask=mask)


def triton_mish_add_clamp_scale(x: torch.Tensor, add_value: float, min_val: float, max_val: float, scale: float):
    """
    Wrapper to launch the Triton kernel.
    """
    assert x.is_cuda, "Input must be a CUDA tensor"
    # Ensure contiguous layout for simple addressing inside Triton kernel
    x_contig = x.contiguous()
    out = torch.empty_like(x_contig)

    n_elements = x_contig.numel()

    # grid based on selected BLOCK_SIZE from autotune
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch kernel
    _fused_mish_add_clamp_scale_kernel[grid](
        x_contig, out, n_elements,
        float(add_value), float(min_val), float(max_val), float(scale)
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that keeps the ConvTranspose2d layer in PyTorch
    but fuses Mish activation, addition, Hardtanh clamp, and scaling
    into a single Triton kernel for improved performance.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        # store scalar params for fused op
        self.add_value = float(add_value)
        self.scale = float(scale)
        # Hardtanh bounds
        self.min_val = -1.0
        self.max_val = 1.0

    def forward(self, x):
        # Perform transposed convolution using PyTorch (leverages cuDNN/cuBLAS)
        x = self.conv_transpose(x)
        # Fuse Mish, add, clamp (Hardtanh), and scale in Triton kernel
        x = triton_mish_add_clamp_scale(x, self.add_value, self.min_val, self.max_val, self.scale)
        return x