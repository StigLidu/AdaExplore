import torch
import torch.nn as nn
import triton
import triton.language as tl

# Triton kernel that computes softmax over the channel dimension (C) for each spatial position
# and then applies sigmoid elementwise to the softmax outputs.
# Assumes input tensor layout is contiguous with shape (N, C, D, H, W).
@triton.jit
def _softmax_sigmoid_kernel(
    x_ptr,            # pointer to input tensor
    out_ptr,          # pointer to output tensor
    num_pos,          # total number of spatial positions = N * D * H * W
    channel_stride,   # stride between channels in number of elements (D*H*W)
    channels,         # number of channels C
    BLOCK_POS: tl.constexpr,  # number of positions processed per program
    BLOCK_C: tl.constexpr     # number of channels processed (expects >= channels)
):
    pid = tl.program_id(0)
    # Offsets of positions this program will handle (each position encodes batch and spatial offset)
    pos_offsets = pid * BLOCK_POS + tl.arange(0, BLOCK_POS)
    pos_mask = pos_offsets < num_pos  # shape (BLOCK_POS,)

    # Compute batch index and offset within channel for each position.
    # channel_stride = D * H * W
    n = pos_offsets // channel_stride                 # shape (BLOCK_POS,)
    offs_in_ch = pos_offsets - n * channel_stride     # shape (BLOCK_POS,)

    # Channel offsets (0..BLOCK_C-1)
    c_offsets = tl.arange(0, BLOCK_C)[:, None]        # shape (BLOCK_C, 1)
    p_offsets = offs_in_ch[None, :]                   # shape (1, BLOCK_POS)

    # Compute batch stride (number of elements to jump to move to next batch)
    batch_stride = channels * channel_stride

    # Build addresses for load: addr = base + n * batch_stride + c * channel_stride + offset_in_channel
    addrs = x_ptr + n[None, :] * batch_stride + c_offsets * channel_stride + p_offsets  # (BLOCK_C, BLOCK_POS)
    mask = (c_offsets < channels) & (pos_mask[None, :])   # shape (BLOCK_C, BLOCK_POS)

    # Load values; out-of-bounds are set to a very small number so max isn't affected
    NEG_INF = tl.constexpr(-1e20)
    vals = tl.load(addrs, mask=mask, other=NEG_INF)  # shape (BLOCK_C, BLOCK_POS)

    # Compute max across channels for numerical stability (per position)
    max_vals = tl.max(vals, axis=0)  # shape (BLOCK_POS,)

    # Subtract max, exponentiate
    vals = vals - max_vals[None, :]
    vals = tl.exp(vals)

    # Sum across channels
    sum_vals = tl.sum(vals, axis=0)  # shape (BLOCK_POS,)

    # Normalize to get softmax, then apply sigmoid
    softmax_vals = vals / (sum_vals[None, :] + 1e-6)
    sigmoid_vals = 1.0 / (1.0 + tl.exp(-softmax_vals))

    # Store results back to output with the same address computation
    out_addrs = out_ptr + n[None, :] * batch_stride + c_offsets * channel_stride + p_offsets
    tl.store(out_addrs, sigmoid_vals, mask=mask)


def triton_softmax_sigmoid(x: torch.Tensor):
    """
    Compute softmax over channel dimension (dim=1) followed by sigmoid,
    for a tensor of shape (N, C, D, H, W) using a Triton kernel.
    Returns a new tensor with the same shape and device as x.
    """
    assert x.is_cuda, "Input must be on CUDA"
    assert x.dtype == torch.float32, "Only float32 is supported"
    x = x.contiguous()
    N, C, D, H, W = x.shape
    # number of spatial positions per batch across all batches
    num_pos = N * D * H * W
    channel_stride = D * H * W  # number of elements to jump to move to next channel

    out = torch.empty_like(x)

    # Choose BLOCK parameters. BLOCK_C must be >= C. Use the actual channel count to avoid mismatch.
    BLOCK_POS = 256  # positions per program
    BLOCK_C = C      # set BLOCK_C equal to actual channels for this launch

    # Grid size: number of programs to cover all positions
    grid = ( (num_pos + BLOCK_POS - 1) // BLOCK_POS, )

    # Launch the kernel. Pass tensors x and out (they are treated as pointers).
    _softmax_sigmoid_kernel[grid](
        x, out,
        num_pos, channel_stride, C,
        BLOCK_POS=BLOCK_POS, BLOCK_C=BLOCK_C
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized Model: uses native ConvTranspose3d for the heavy transpose convolution,
    but fuses Softmax(dim=1) + Sigmoid into a single Triton kernel for better performance.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super(ModelNew, self).__init__()
        # Keep the original ConvTranspose3d (efficient cuDNN implementation)
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias)

    def forward(self, x):
        # Transposed convolution (leave to PyTorch/cuDNN)
        x = self.conv_transpose(x)
        # Fuse Softmax(dim=1) followed by Sigmoid using Triton kernel
        x = triton_softmax_sigmoid(x)
        return x


# Keep helper functions similar to original for compatibility
batch_size = 16
in_channels = 32
out_channels = 64
D, H, W = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding]