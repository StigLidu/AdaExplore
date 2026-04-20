import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configs for the Triton kernel
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 512}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=4),
]


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["n_elements"])
@triton.jit
def _bias_sub_tanh_kernel(
    x_ptr,        # pointer to input/output tensor (N*C*H*W elements)
    bias_ptr,     # pointer to bias per channel (C elements)
    out_ptr,      # pointer to output tensor
    N, C, H, W,   # tensor dimensions
    HW, CHW,      # precomputed H*W and C*H*W
    n_elements,   # total number of elements
    BLOCK: tl.constexpr,
):
    """
    For each flattened index `idx` in [0, n_elements):
      - Determine channel c = (idx % (C*H*W)) // (H*W)
      - Load x[idx], bias[c], compute x - bias[c], apply tanh.
    The tanh is computed via a numerically stable expression using exp.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK
    offs = block_start + tl.arange(0, BLOCK)
    mask = offs < n_elements

    # Load x values (fp32)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # Compute per-element channel index:
    # rem = offs % (C*H*W)
    # rem = offs - (offs // CHW) * CHW
    offs_div_chw = offs // CHW
    rem = offs - offs_div_chw * CHW
    # c = rem // HW
    c = rem // HW

    # Load bias for each element's channel
    bias_vals = tl.load(bias_ptr + c, mask=mask, other=0.0)

    # Compute x - bias
    y = x - bias_vals

    # Compute tanh(y) via exp to avoid relying on tl.tanh
    # tanh(y) = sign(y) * (1 - exp(-2*|y|)) / (1 + exp(-2*|y|))
    a = tl.abs(y)
    e = tl.exp(-2.0 * a)
    num = 1.0 - e
    den = 1.0 + e
    tanh_pos = num / den
    # restore sign
    res = tl.where(y >= 0.0, tanh_pos, -tanh_pos)

    # Store result
    tl.store(out_ptr + offs, res, mask=mask)


def triton_bias_sub_tanh(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Fuses: x - bias (broadcast over channels) followed by tanh, using Triton kernel.
    x: tensor of shape (N, C, H, W), dtype float32, on CUDA
    bias: tensor of shape (C, 1, 1) or (C,) broadcastable to channels. on CUDA
    """
    assert x.is_cuda and bias.is_cuda, "x and bias must be on CUDA"
    assert x.dtype == torch.float32 and bias.dtype == torch.float32, "Only float32 supported"
    x = x.contiguous()
    # Prepare bias as 1D contiguous per-channel array
    bias_1d = bias.view(-1).contiguous()

    N, C, H, W = x.shape
    n_elements = x.numel()
    HW = H * W
    CHW = C * HW

    out = torch.empty_like(x)

    # grid: number of blocks
    def grid(meta):
        BLOCK = meta["BLOCK"]
        return ((n_elements + BLOCK - 1) // BLOCK,)

    # Launch kernel
    _bias_sub_tanh_kernel[grid](
        x,
        bias_1d,
        out,
        N, C, H, W,
        HW, CHW,
        n_elements,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized Model: uses PyTorch ConvTranspose2d for the heavy convolution transpose,
    and a fused Triton kernel to subtract the per-channel bias and apply tanh in one pass.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        # Keep the standard ConvTranspose2d (highly optimized in cuDNN/CUDA)
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        # Bias stored same as original: shape (out_channels, 1, 1)
        self.bias = nn.Parameter(torch.randn(bias_shape, dtype=torch.float32))

    def forward(self, x):
        # Use existing efficient implementation for conv_transpose
        x = self.conv_transpose(x)
        # Apply fused bias-subtraction + tanh via Triton kernel
        # Ensure bias is same device/dtype
        bias = self.bias
        if bias.device != x.device:
            bias = bias.to(x.device)
        return triton_bias_sub_tanh(x, bias)