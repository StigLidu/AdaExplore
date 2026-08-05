import torch
import torch.nn as nn
import triton
import triton.language as tl

# Triton kernel: fused logsumexp over channel dim + "HardSwish-like" op (x * sigmoid(x+3)/6) + subtract bias + clamp
# Kernel expects input arranged as (C, total) contiguous, where total = B * D * H * W
@triton.jit()
def fused_lse_hs_kernel(
    x_ptr,          # pointer to input tensor shaped (B, C, D, H, W) flattened as 1D (we index via channel stride)
    out_ptr,        # pointer to output tensor shaped (total,)
    total,          # total = B * D * H * W
    bias,           # scalar bias (float)
    channel_stride, # stride (in elements) between channels in the input tensor (x.stride()[1])
    BLOCK: tl.constexpr,
    CHANNELS: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK
    offs = block_start + tl.arange(0, BLOCK)
    mask = offs < total

    # acc_max initialized to very small number for numeric stability
    acc_max = tl.full((BLOCK,), -1e20, dtype=tl.float32)

    # First pass: compute max across channels
    for k in range(CHANNELS):
        # address for channel k using channel stride passed from host
        addrs = k * channel_stride + offs
        vals = tl.load(x_ptr + addrs, mask=mask, other=-1e20)
        acc_max = tl.maximum(acc_max, vals)

    # Second pass: compute sum exp(x - max)
    sumexp = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in range(CHANNELS):
        addrs = k * channel_stride + offs
        vals = tl.load(x_ptr + addrs, mask=mask, other=0.0)
        # compute delta and clamp to <= 0.0 to avoid exp overflow on masked/invalid lanes
        delta = vals - acc_max
        delta = tl.minimum(delta, 0.0)
        sumexp = sumexp + tl.exp(delta)

    # logsumexp
    lse = acc_max + tl.log(sumexp)

    # Hard-swish like: x * sigmoid(x + 3) / 6  (sigmoid implemented as 1/(1+exp(-z)))
    z = lse + 3.0
    sig = 1.0 / (1.0 + tl.exp(-z))
    out = lse * sig / 6.0

    # subtract bias (broadcasted) and clamp to [-1, 1]
    out = out - bias
    out = tl.maximum(out, -1.0)
    out = tl.minimum(out, 1.0)

    # store result
    tl.store(out_ptr + offs, out, mask=mask)


def triton_fused_logsumexp_hs(x: torch.Tensor, bias_scalar: float):
    """
    x: Tensor of shape (B, C, D, H, W) on CUDA, dtype float32
    returns: Tensor of shape (B, 1, D, H, W) on same device/dtype
    """
    assert x.is_cuda, "Input must be on CUDA"
    B, C, D, H, W = x.shape
    total = B * D * H * W
    # Use the input tensor's channel stride to index channels without a costly permute/contiguous
    channel_stride = x.stride()[1]

    # Prepare output flat tensor
    out_flat = torch.empty((total,), device=x.device, dtype=x.dtype)

    # Kernel launch parameters (tuning knobs)
    BLOCK = 1024
    grid = ( (total + BLOCK - 1) // BLOCK, )

    # Launch Triton kernel. Pass channel_stride and tune num_warps at launch time.
    fused_lse_hs_kernel[grid](x, out_flat, total, float(bias_scalar), channel_stride, BLOCK=BLOCK, CHANNELS=C, num_warps=4)

    # reshape to (B, 1, D, H, W)
    out = out_flat.view(B, 1, D, H, W).contiguous()
    return out


class ModelNew(nn.Module):
    """
    Optimized model: keeps the ConvTranspose3d in PyTorch, but fuses the following sequence:
    logsumexp over channels -> x * sigmoid(x + 3) / 6 -> subtract bias -> clamp into a single Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        # Use the same ConvTranspose3d as original for correctness and performance (cuDNN/ATen)
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        # bias kept as a parameter (shape compatible for broadcasting)
        self.bias = nn.Parameter(torch.randn(*bias_shape))

    def forward(self, x):
        # x: (B, in_channels, D, H, W)
        x = self.conv_transpose(x)  # -> (B, out_channels, D', H', W')
        # Fused post-processing via Triton kernel
        # Ensure bias is a scalar for our fused kernel; original bias shape is broadcastable
        # Use the first element as the scalar bias to subtract
        bias_scalar = float(self.bias.view(-1)[0])
        out = triton_fused_logsumexp_hs(x, bias_scalar)
        return out


# Re-create helper functions similar to the original module to ease integration.
batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
bias_shape = (1, 1, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias_shape]