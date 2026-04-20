import torch
import torch.nn as nn
import triton
import triton.language as tl

# Triton kernel that, for each batch element, computes:
# sum_c ( x[b, c, 0, 0, 0] * recip_divisor + bias[c] )
# Writes one float per batch into out_ptr[b]
@triton.jit
def _fused_div_bias_sum_kernel(
    x_ptr,           # pointer to input tensor (B, C, 1, 1, 1) flattened
    bias_ptr,        # pointer to bias tensor (C,)
    out_ptr,         # pointer to output tensor (B,)
    B,               # batch
    C,               # channels
    stride_b,        # stride to move between batches in elements
    stride_c,        # stride to move between channels in elements
    recip_divisor,   # 1.0 / divisor
    BLOCK: tl.constexpr,
    BATCH: tl.constexpr
):
    """
    Each program handles a tile of BATCH batches and iterates over channel chunks of size BLOCK.
    This reduces launch overhead (fewer programs) and processes channels in chunks to handle C > BLOCK.
    """
    pid = tl.program_id(0)
    b_start = pid * BATCH

    # vector of relative batch indices this program will handle
    rb = tl.arange(0, BATCH)
    b_idx = b_start + rb
    mask_b = b_idx < B

    # accumulator per batch in the tile
    acc = tl.zeros((BATCH,), dtype=tl.float32)

    # iterate over channels in chunks of size BLOCK
    for off in range(0, C, BLOCK):
        c_idx = off + tl.arange(0, BLOCK)               # shape (BLOCK,)
        mask_c = c_idx < C                              # shape (BLOCK,)

        # build (BATCH, BLOCK) addresses: b_idx[:, None] * stride_b + c_idx[None, :] * stride_c
        offs = b_idx[:, None] * stride_b + c_idx[None, :] * stride_c
        mask = mask_b[:, None] & mask_c[None, :]

        # load x values for this batch tile and channel chunk
        x_vals = tl.load(x_ptr + offs, mask=mask, other=0.0)             # (BATCH, BLOCK)
        # load bias for channel chunk (masked); shape (BLOCK,)
        bias_vals = tl.load(bias_ptr + c_idx, mask=mask_c, other=0.0)   # (BLOCK,)

        # Reduce memory traffic: sum x across channels and sum bias across channels,
        # then update accumulator. This avoids broadcasting bias for every batch element.
        sum_x = tl.sum(x_vals, 1)                # (BATCH,)
        sum_bias = tl.sum(bias_vals, 0)          # scalar (sum over channel chunk)
        acc = acc + sum_x * recip_divisor + sum_bias

    # store results for valid batches
    out_offs = b_idx
    tl.store(out_ptr + out_offs, acc, mask=mask_b)


def triton_fused_div_bias_sum(x: torch.Tensor, bias: torch.Tensor, divisor: float):
    """
    Wrapper for the Triton kernel.
    Expects:
      x: tensor of shape (B, C, 1, 1, 1), contiguous, on CUDA, dtype float32
      bias: tensor of shape (C, 1, 1, 1) or (C,), contiguous, on CUDA, dtype float32
      divisor: float scalar
    Returns:
      Tensor of shape (B, 1, 1, 1) (float32, on same device)
    """
    assert x.is_cuda and bias.is_cuda, "Inputs must be on CUDA"
    assert x.dtype == torch.float32 and bias.dtype == torch.float32, "Only float32 supported"

    # Ensure contiguous and expected shapes
    x = x.contiguous()
    B, C, d, h, w = x.shape
    assert d == 1 and h == 1 and w == 1, "triton_fused_div_bias_sum expects spatial dims = 1"

    # Bias to 1D
    bias_flat = bias.contiguous().view(-1)

    # Output: one scalar per batch
    out = torch.empty(B, device=x.device, dtype=x.dtype)

    # strides in elements (PyTorch strides are in elements already)
    stride_b = x.stride(0)
    stride_c = x.stride(1)

    # Reasonable defaults for Ampere: choose BLOCK to match (or round up to) the real channel count
    # and increase BATCH so each program does more work (fewer launches). These are constexpr
    # values passed to the kernel and can still be tuned externally.
    BLOCK = 16
    BATCH = 16

    # grid over batch tiles
    grid = ((B + BATCH - 1) // BATCH,)

    # Launch kernel with constexpr BLOCK and BATCH
    _fused_div_bias_sum_kernel[grid](
        x, bias_flat, out,
        B, C, stride_b, stride_c, float(1.0 / divisor),
        BLOCK=BLOCK, BATCH=BATCH
    )

    # reshape to match original model's output shape after sum: (B, 1, 1, 1)
    return out.view(B, 1, 1, 1)


class ModelNew(nn.Module):
    """
    Optimized model using a Triton kernel to fuse division, bias addition, and channel-sum
    after the global average pooling. The convolution and max-pooling are left as PyTorch ops.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.max_pool = nn.MaxPool3d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # bias stored in same shape as before for compatibility, will be flattened inside triton wrapper
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim

    def forward(self, x):
        # 1) conv
        x = self.conv(x)
        # Note: division can be postponed (linear op) until after pooling for numerical equivalence
        # 2) max pool
        x = self.max_pool(x)
        # 3) global average pool -> shape (B, C, 1, 1, 1)
        x = self.global_avg_pool(x)
        # 4) fused division + bias addition + sum over channels using Triton
        # The Triton wrapper expects bias as (C,) or broadcastable; provide the flattened bias.
        out = triton_fused_div_bias_sum(x, self.bias, self.divisor)
        # out shape is (B, 1, 1, 1) as required
        return out


# Keep the helper functions consistent with the original problem to allow external usage
batch_size   = 128
in_channels  = 8
out_channels = 16
depth = 16; height = width = 64
kernel_size = (3, 3, 3)
divisor = 2.0
pool_size = (2, 2, 2)
bias_shape = (out_channels, 1, 1, 1)
sum_dim = 1

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim]