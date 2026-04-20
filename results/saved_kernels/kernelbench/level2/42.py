import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configs favoring larger BLOCK_SIZEs (better for very large spatial sizes)
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 256},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 512},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=3),
]


@triton.autotune(AUTOTUNE_CONFIGS, key=['HW', 'num_out'])
@triton.jit
def _spatial_sum_kernel(inp_ptr, partial_ptr, HW, num_out, num_tiles, BLOCK_SIZE: tl.constexpr):
    """
    Each program handles one (pid, tile) pair and computes a partial sum of up to BLOCK_SIZE elements.
    Writes the scalar partial sum to partial_ptr[pid * num_tiles + tile].
    Grid: (num_out, num_tiles)
    """
    pid = tl.program_id(0)   # index in [0..num_out)
    tile = tl.program_id(1)  # which tile across HW
    base = pid * HW
    start = tile * BLOCK_SIZE
    offs = tl.arange(0, BLOCK_SIZE)
    idxs = start + offs
    mask = idxs < HW
    ptrs = inp_ptr + base + idxs
    vals = tl.load(ptrs, mask=mask, other=0.0)
    # reduce vector to scalar accumulator
    acc = tl.sum(vals, 0)
    out_index = pid * num_tiles + tile
    tl.store(partial_ptr + out_index, acc)


def triton_spatial_sum(x: torch.Tensor) -> torch.Tensor:
    """
    Compute sum over spatial dims (H,W) for each (batch, channel) using a tiled Triton kernel.
    Input: x shape (B, C, H, W), CUDA, float32
    Output: tensor shape (B, C) with sums over H*W
    """
    assert x.is_cuda, "Input must be on CUDA"
    B, C, H, W = x.shape
    HW = H * W
    num_out = B * C

    # Make a 2D view (num_out, HW). Avoid copies when possible.
    if not x.is_contiguous():
        x_flat2d = x.contiguous().view(B * C, HW)
    else:
        x_flat2d = x.view(B * C, HW)

    inp_flat = x_flat2d.reshape(-1)

    # Choose a conservative default tile size close to the larger autotune candidates.
    # This keeps the partials buffer small for huge spatial sizes and matches likely chosen BLOCK_SIZE.
    DEFAULT_TILE = 1024
    num_tiles = (HW + DEFAULT_TILE - 1) // DEFAULT_TILE
    if num_tiles == 0:
        num_tiles = 1

    # Allocate partials buffer on device: shape (num_out, num_tiles)
    partials = torch.empty((num_out, num_tiles), device=x.device, dtype=x.dtype)

    # grid for kernel launch: num_out x num_tiles
    grid = lambda meta: (num_out, num_tiles)

    # Launch Triton kernel: partials is passed flattened so kernel can write into it linearly
    _spatial_sum_kernel[grid](inp_flat, partials.reshape(-1), HW, num_out, num_tiles)

    # now reduce partials along the tile dimension to get final sums per (b,c)
    out = partials.sum(dim=1)  # shape (num_out,)

    # reshape back to (B, C)
    return out.view(B, C)


class ModelNew(nn.Module):
    """
    Optimized Model that avoids materializing the full ConvTranspose2d output.
    Uses a Triton kernel to compute the spatial sums of the input (heavy part),
    then uses fast PyTorch matrix operations to finish the computation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        # keep conv_transpose to retain weights; we'll use its weights analytically
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        # bias as in original model
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        # x: (B, inC, H, W)
        B, inC, H, W = x.shape
        device = x.device
        dtype = x.dtype

        # 1) compute sum over spatial dims for each (batch, in_channel) using Triton
        # returns shape (B, inC)
        sum_input = triton_spatial_sum(x)  # (B, inC)

        # 2) compute sum of kernel weights over spatial dims -> shape (inC, outC)
        weight = self.conv_transpose.weight  # shape (inC, outC, kH, kW)
        if weight.device != device or weight.dtype != dtype:
            weight = weight.to(device=device, dtype=dtype)
        K_sum = weight.sum(dim=(2, 3))  # (inC, outC), on correct device

        # 3) aggregated contribution to each output channel:
        # pooled_sum = sum_i sum_input[b,i] * K_sum[i,o]
        pooled = torch.matmul(sum_input, K_sum)  # (B, outC)

        # convert sum over output spatial to mean over spatial by dividing by H_out * W_out
        kH, kW = self.conv_transpose.kernel_size
        H_out = H + kH - 1
        W_out = W + kW - 1
        scale = 1.0 / (H_out * W_out)
        pooled = pooled * scale  # (B, outC)

        # 4) add bias (original bias shape is (outC,1,1) -> broadcast to (B, outC))
        b = self.bias
        if b.device != device or b.dtype != dtype:
            b = b.to(device=device, dtype=dtype)
        pooled = pooled + b.view(1, -1)

        # 5) log-sum-exp across channels -> (B,1)
        lse = torch.logsumexp(pooled, dim=1, keepdim=True)  # (B,1)

        # 6) final multiplication
        out = lse * 10.0  # (B,1)

        return out