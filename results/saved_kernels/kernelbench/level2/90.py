import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configurations chosen for A6000 (Ampere)
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=3),
]

@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['N', 'C', 'spatial_size'])
@triton.jit
def fused_postprocess_kernel(
    x_ptr,        # pointer to input tensor (flattened)
    sum_ptr,      # pointer to per-channel sum (length C)
    out_ptr,      # pointer to output tensor (flattened)
    N,            # batch size
    C,            # number of channels
    spatial_size, # D*H*W
    BLOCK_SIZE: tl.constexpr,
):
    """
    Each program processes a contiguous block of size BLOCK_SIZE from a single (n, c) channel slice.
    Mapping:
      pid = program id in [0 .. N*C*blocks_per_channel-1]
      n = pid // (C * blocks_per_channel)
      rem = pid % (C * blocks_per_channel)
      c = rem // blocks_per_channel
      s_block = rem % blocks_per_channel
    start_offset = ((n * C + c) * spatial_size) + s_block * BLOCK_SIZE
    """
    pid = tl.program_id(0)

    # Compute blocks_per_channel from constexpr BLOCK_SIZE and runtime spatial_size
    blocks_per_channel = (spatial_size + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Compute n, c, and spatial block index
    channel_block_span = C * blocks_per_channel
    n = pid // channel_block_span
    rem = pid - n * channel_block_span
    c = rem // blocks_per_channel
    s_block = rem - c * blocks_per_channel

    # Compute the starting flattened offset for this (n, c, s_block)
    start = (n * C + c) * spatial_size + s_block * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    # mask for spatial bounds within this channel
    max_off = (n * C + c) * spatial_size + spatial_size
    mask = offs < max_off

    # Load input values
    x_vals = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # Load the per-channel addend (scalar) and broadcast via arithmetic
    sum_val = tl.load(sum_ptr + c)  # scalar load

    # LeakyReLU (negative_slope=0.2)
    neg_slope = 0.2
    x_relu = tl.where(x_vals > 0.0, x_vals, x_vals * neg_slope)

    # Add per-channel value
    x_added = x_relu + sum_val

    # Clamp to [-1.0, 1.0]
    x_clamped = tl.where(x_added < -1.0, -1.0, tl.where(x_added > 1.0, 1.0, x_added))

    # GELU approx: x * sigmoid(1.702 * x)
    k = 1.702
    sig = 1.0 / (1.0 + tl.exp(-k * x_clamped))
    y = x_clamped * sig

    # Store result
    tl.store(out_ptr + offs, y, mask=mask)


def triton_fused_postprocess(x: torch.Tensor, sum_tensor: torch.Tensor) -> torch.Tensor:
    """
    Wrapper that launches the Triton fused kernel.
    Assumes x is contiguous CUDA float32 tensor with shape [N, C, D, H, W].
    sum_tensor is of shape [C] or [C,1,1,1] (will be flattened).
    """
    assert x.is_cuda and sum_tensor.is_cuda, "Tensors must be on CUDA"
    assert x.dtype == torch.float32 and sum_tensor.dtype == torch.float32

    x = x.contiguous()
    sum_flat = sum_tensor.reshape(-1).contiguous()
    out = torch.empty_like(x)

    N, C, D, H, W = x.shape
    spatial_size = D * H * W

    # grid function uses the autotuned BLOCK_SIZE from meta to compute the number of programs
    def grid(meta):
        BLOCK = meta["BLOCK_SIZE"]
        bpc = (spatial_size + BLOCK - 1) // BLOCK
        total_blocks = N * C * bpc
        return (total_blocks,)

    # Launch the autotuned kernel. The kernel computes blocks_per_channel internally from BLOCK_SIZE.
    fused_postprocess_kernel[grid](
        x,
        sum_flat,
        out,
        N,
        C,
        spatial_size
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized model: keep PyTorch Conv3d for correctness and use a Triton-fused kernel
    to perform the post-convolution elementwise chain:
      - LeakyReLU(negative_slope=0.2)
      - Add per-channel sum_tensor (broadcast)
      - Clamp to [-1, 1]
      - GELU (approx: x * sigmoid(1.702*x))
    The Triton kernel processes blocks of spatial elements per (N, C) slice to minimize repeated loads of the per-channel addend.
    """
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))

    def forward(self, x):
        x = self.conv(x)
        # Ensure tensors are on CUDA when calling the Triton wrapper
        if x.is_cuda and self.sum_tensor.is_cuda:
            x = triton_fused_postprocess(x, self.sum_tensor)
        else:
            # CPU or mismatched device: fall back to pure PyTorch implementation
            x = torch.nn.functional.leaky_relu(x, negative_slope=0.2)
            x = x + self.sum_tensor
            x = torch.clamp(x, min=-1.0, max=1.0)
            x = torch.nn.functional.gelu(x)
        return x


# Compatibility helper functions (same signatures as original)
def get_inputs():
    batch_size = 128
    in_channels = 8
    depth, height, width = 16, 64, 64
    # Return CUDA tensor for best performance with the Triton kernel
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    in_channels = 8
    out_channels = 64
    kernel_size = 3
    sum_tensor_shape = (out_channels, 1, 1, 1)
    return [in_channels, out_channels, kernel_size, sum_tensor_shape]