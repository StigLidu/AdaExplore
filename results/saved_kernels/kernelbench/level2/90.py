import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configurations for the fused kernel. BLOCK is the number of spatial elements processed per program.
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 128}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 512}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=3),
]

@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['B', 'C', 'S'])
@triton.jit
def fused_pointwise_kernel(
    x_ptr,         # pointer to conv output (flattened)
    sum_ptr,       # pointer to sum tensor (length C)
    out_ptr,       # pointer to output (flattened)
    B,             # batch size
    C,             # channels
    S,             # spatial size per channel: D*H*W
    neg_slope,     # leaky relu negative slope (float)
    BLOCK: tl.constexpr
):
    """
    Grid layout:
      program_id(0) -> batch index (0..B-1)
      program_id(1) -> channel index (0..C-1)
      program_id(2) -> spatial block index (0..ceil(S/BLOCK)-1)
    Each program processes up to BLOCK spatial elements for a given (batch,channel).
    """
    b = tl.program_id(0)
    c = tl.program_id(1)
    block_idx = tl.program_id(2)

    offs = block_idx * BLOCK + tl.arange(0, BLOCK)
    mask = offs < S

    # base offset for this (b,c)
    # linear index = ((b * C) + c) * S + offs
    base = ((b * C) + c) * S
    idx = base + offs

    # load input (masked)
    x = tl.load(x_ptr + idx, mask=mask, other=0.0)

    # load per-channel bias/addition
    sum_v = tl.load(sum_ptr + c)

    # leaky relu: if x >= 0 -> x else x * neg_slope
    x = tl.where(x >= 0.0, x, x * neg_slope)

    # add the per-channel tensor (broadcast)
    x = x + sum_v

    # clamp between -1 and 1
    x = tl.where(x < -1.0, -1.0, x)
    x = tl.where(x >  1.0,  1.0, x)

    # GELU approximation using sigmoid: x * sigmoid(1.702 * x)
    y = 1.702 * x
    sig = 1.0 / (1.0 + tl.exp(-y))
    x = x * sig

    # store results
    tl.store(out_ptr + idx, x, mask=mask)


def triton_fused_pointwise(x: torch.Tensor, sum_tensor: torch.Tensor, neg_slope: float = 0.2):
    """
    Wrapper to launch the Triton fused kernel.
    Expects:
      x: conv output tensor of shape [B, C, D, H, W], contiguous on CUDA
      sum_tensor: tensor of shape [C, 1, 1, 1] or [C], contiguous on CUDA
    Returns:
      out tensor with same shape as x
    """
    assert x.is_cuda and sum_tensor.is_cuda, "Tensors must be on CUDA."
    assert x.dtype == torch.float32 and sum_tensor.dtype == torch.float32, "Only fp32 supported."

    # ensure contiguous
    x = x.contiguous()
    # make sum_tensor a 1D contiguous tensor of length C
    sum_1d = sum_tensor.view(sum_tensor.shape[0]).contiguous()

    B, C, D, H, W = x.shape
    S = D * H * W

    out = torch.empty_like(x)

    # flatten tensors for pointer arithmetic in the kernel
    x_flat = x.view(-1)
    out_flat = out.view(-1)
    sum_flat = sum_1d

    # grid: (B, C, number of spatial blocks)
    def grid(meta):
        return (B, C, (S + meta['BLOCK'] - 1) // meta['BLOCK'])

    fused_pointwise_kernel[grid](x_flat, sum_flat, out_flat, B, C, S, float(neg_slope))
    return out


class ModelNew(nn.Module):
    """
    Optimized model that uses PyTorch's Conv3d but fuses the subsequent
    elementwise operations (LeakyReLU, add with sum_tensor, clamp, GELU)
    into a single Triton kernel for improved performance.
    """
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        # sum_tensor expected shape: (out_channels, 1, 1, 1)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape, dtype=torch.float32))

    def forward(self, x):
        # conv uses PyTorch implementation (GPU)
        x = self.conv(x)
        # fused pointwise operations implemented in Triton
        x = triton_fused_pointwise(x, self.sum_tensor, neg_slope=0.2)
        return x