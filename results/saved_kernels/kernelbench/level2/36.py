import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Triton kernel that, for each (batch, w-block), computes:
#   for each h-block:
#     min over channels for the (h_block x w_block) tile
#     sum over h in that tile -> partial sums per w
#   accumulate partial sums across h_blocks -> final sum over H of min_c X[b,c,h,w]
# This kernel avoids atomic adds by making each program handle ALL H blocks for its (b, w_block).
@triton.jit
def _minsum_kernel(
    x_ptr,               # input pointer (fp32)
    out_ptr,             # output pointer (fp32) shape [B, 1, 1, W] linearized as b*W + w
    B, C, H, W,          # tensor dims
    stride_batch,        # elements between batches = C*H*W
    stride_channel,      # elements between channels = H*W
    stride_h,            # elements between h positions = W
    num_h_blocks,        # number of h blocks (runtime int)
    BLOCK_H: tl.constexpr,   # height block size (constexpr)
    BLOCK_W: tl.constexpr,   # width block size (constexpr)
    BLOCK_C: tl.constexpr,   # channel block size (constexpr)
):
    # program ids: program_id(0)=batch, program_id(1)=w_block, program_id(2)=h_block
    b = tl.program_id(0)
    w_block = tl.program_id(1)
    hb = tl.program_id(2)

    # width indices handled by this program
    w_start = w_block * BLOCK_W
    offs_w = tl.arange(0, BLOCK_W)                 # (BLOCK_W,)
    w_idx = w_start + offs_w                       # (BLOCK_W,)
    mask_w = w_idx < W                             # (BLOCK_W,)

    # height indices handled by this program (one h-block per program)
    h_start = hb * BLOCK_H
    offs_h = tl.arange(0, BLOCK_H)                 # (BLOCK_H,)
    h_idx = h_start + offs_h                       # (BLOCK_H,)
    mask_h = h_idx < H                             # (BLOCK_H,)

    # h and w offsets for addressing
    h_offsets = h_idx * stride_h                   # (BLOCK_H,)
    w_offsets = w_idx                              # (BLOCK_W,)

    # Initialize min accumulator for this (h_block x w_block)
    # Shape: (BLOCK_H, BLOCK_W)
    min_vals = tl.full((BLOCK_H, BLOCK_W), 1e9, dtype=tl.float32)

    # Loop over channels in blocks (each program handles its own h-block)
    c = 0
    while c < C:
        for c_inner in range(BLOCK_C):
            c_idx = c + c_inner
            mask_c = c_idx < C  # scalar boolean

            # base addresses: b * stride_batch + c_idx * stride_channel + h_offsets[:,None] + w_offsets[None,:]
            base = b * stride_batch + c_idx * stride_channel + h_offsets[:, None] + w_offsets[None, :]

            # Combined mask: valid h, valid w, valid c
            load_mask = mask_h[:, None] & mask_w[None, :] & mask_c

            # Load tile and update min accumulator
            vals = tl.load(x_ptr + base, mask=load_mask, other=1e9)
            min_vals = tl.minimum(min_vals, vals)
        c += BLOCK_C

    # After processing channels for this h-block, sum min over H dimension (ignoring OOB h lanes)
    valid_min = tl.where(mask_h[:, None], min_vals, 0.0)  # zero out OOB h rows
    sum_over_h_block = tl.sum(valid_min, 0)               # -> shape (BLOCK_W,)

    # Atomic add partial sums into the output (multiple programs add into the same w entries)
    out_base = b * W + w_idx  # (BLOCK_W,)
    tl.atomic_add(out_ptr + out_base, sum_over_h_block, mask=mask_w)


def triton_min_sum(x: torch.Tensor, BLOCK_H: int = 8, BLOCK_W: int = 64, BLOCK_C: int = 16):
    """
    Wrapper to launch the Triton kernel.
    x: [B, C, H, W] contiguous cuda float32
    returns: tensor of shape [B, 1, 1, W] (cuda float32)
    """
    assert x.is_cuda, "Input must be CUDA tensor"
    assert x.dtype == torch.float32, "Only float32 supported"
    x = x.contiguous()
    B, C, H, W = x.shape

    # output accumulator [B,1,1,W] as a linear array per b,w
    out = torch.empty((B, 1, 1, W), device=x.device, dtype=x.dtype)
    out.zero_()

    stride_batch = C * H * W
    stride_channel = H * W
    stride_h = W

    num_h_blocks = (H + BLOCK_H - 1) // BLOCK_H
    num_w_blocks = (W + BLOCK_W - 1) // BLOCK_W

    # grid: (B, num_w_blocks, num_h_blocks) -> more parallelism
    grid = (B, num_w_blocks, num_h_blocks)

    _minsum_kernel[grid](
        x,
        out.view(-1),  # we'll treat out as linearized [B*W]
        B, C, H, W,
        stride_batch,
        stride_channel,
        stride_h,
        num_h_blocks,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
        BLOCK_C=BLOCK_C,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model:
      - Keep ConvTranspose2d in PyTorch (well-optimized).
      - Use a fused Triton kernel that computes channel-wise minimum and sums over height,
        avoiding atomics by letting each program cover all H blocks for its (batch, w-block).
      - Apply GELU and bias addition in PyTorch (these are cheap compared to the reduction).
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )
        # bias is shape (1,1,1) as in original example; will broadcast to [B,1,1,W]
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        # conv transpose (PyTorch)
        x = self.conv_transpose(x)  # [B, C, H', W']
        # fused Triton kernel: min over C, sum over H -> [B,1,1,W']
        s = triton_min_sum(x, BLOCK_H=8, BLOCK_W=64, BLOCK_C=16)
        # GELU and bias (PyTorch)
        s = F.gelu(s)
        s = s + self.bias
        return s


# Retain helpers similar to original for initialization/testing
batch_size = 16
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (1, 1, 1)


def get_inputs():
    # return CUDA tensor since Triton kernel requires CUDA inputs
    return [torch.rand(batch_size, in_channels, height, width, device='cuda', dtype=torch.float32)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]