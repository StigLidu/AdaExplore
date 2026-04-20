import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

# Autotune configurations for the Triton kernel
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 256}, num_warps=8, num_stages=2),
]


@triton.autotune(
    configs=AUTOTUNE_CONFIGS,
    key=['N', 'C', 'D_out', 'H_out', 'W_out'],
)
@triton.jit
def fused_avgpool3d_kernel(
    inp_ptr,           # pointer to input tensor (N, C, D, H, W)
    out_ptr,           # pointer to output tensor (N, C, D_out, H_out, W_out)
    N, C, D, H, W,     # input dims
    D_out, H_out, W_out,  # output dims (expected D_out = D // 4, etc.)
    BLOCK: tl.constexpr
):
    """
    This kernel computes the effect of two successive AvgPool3d(kernel_size=2, stride=2)
    operations by averaging over 4x4x4 blocks of the input tensor. Only full 4-blocks
    are processed; this matches the behavior of two sequential non-overlapping kernel=2 pools
    (i.e., final output dims are floor(D/4), floor(H/4), floor(W/4)).
    Grid dimensions:
      - program_id(0): iterates over (N * C * D_out * H_out)
      - program_id(1): iterates over blocks covering W_out with BLOCK elements per program
    """
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    # Decompose pid0 into n, c, dz_out, dy_out
    # total per n = C * D_out * H_out
    total_per_n = C * D_out * H_out
    n = pid0 // total_per_n
    rem0 = pid0 % total_per_n
    c = rem0 // (D_out * H_out)
    rem1 = rem0 % (D_out * H_out)
    dz = rem1 // H_out
    dy = rem1 % H_out

    # w positions handled by this program (vector)
    w_idx = pid1 * BLOCK + tl.arange(0, BLOCK)
    mask_w = w_idx < W_out

    # compute base coordinates in the input (start of the 4x4x4 block)
    z_base = dz * 4
    y_base = dy * 4
    x_base = w_idx * 4  # vector of starting x indices for each output position handled

    # Initialize accumulator
    acc = tl.zeros((BLOCK,), dtype=tl.float32)

    # Sum over 4x4x4 block
    # Note: For valid outputs (w_idx < W_out, dy < H_out, dz < D_out) the full 4x4x4 block is inside input bounds
    for dz_i in range(4):
        z = z_base + dz_i  # scalar
        for dy_i in range(4):
            y = y_base + dy_i  # scalar
            # For the x dimension we have a vector x_base + dx
            for dx_i in range(4):
                x = x_base + dx_i  # vector of x indices
                # Compute flattened input offsets:
                # offset = ((((n * C + c) * D + z) * H + y) * W) + x
                offs = (((n * C + c) * D + z) * H + y) * W + x
                vals = tl.load(inp_ptr + offs, mask=mask_w, other=0.0)
                acc += vals

    # Average over 64 elements (4*4*4)
    out_vals = acc * (1.0 / 64.0)

    # Compute output flattened offsets: ((((n * C + c) * D_out + dz) * H_out + dy) * W_out) + w_idx
    out_offs = (((n * C + c) * D_out + dz) * H_out + dy) * W_out + w_idx
    tl.store(out_ptr + out_offs, out_vals, mask=mask_w)


def triton_fused_avgpool3d(x: torch.Tensor):
    """
    Wrapper that runs the fused 2x sequential avg pooling (kernel=2, stride=2 twice)
    as a single 4x4x4 avg pooling over complete blocks using a Triton kernel.
    """
    assert x.is_cuda, "Input must be on CUDA"
    assert x.dtype == torch.float32, "This fused kernel expects fp32 inputs"
    x = x.contiguous()
    N, C, D, H, W = x.shape

    # Output dims: only full 4-blocks are considered (same behavior as two non-overlapping 2x pools)
    D_out = D // 4
    H_out = H // 4
    W_out = W // 4

    if D_out == 0 or H_out == 0 or W_out == 0:
        # Fallback to PyTorch (input too small for fused 4-block pooling)
        x1 = nn.AvgPool3d(kernel_size=2)(x)
        x2 = nn.AvgPool3d(kernel_size=2)(x1)
        return x2

    out = x.new_empty((N, C, D_out, H_out, W_out), dtype=torch.float32)

    # Grid: (N * C * D_out * H_out, ceil_div(W_out, BLOCK))
    def grid(meta):
        return (N * C * D_out * H_out, (W_out + meta['BLOCK'] - 1) // meta['BLOCK'])

    fused_avgpool3d_kernel[grid](
        x, out,
        N, C, D, H, W,
        D_out, H_out, W_out
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that uses the original ConvTranspose3d and BatchNorm3d,
    but replaces the two sequential AvgPool3d(kernel_size=2) calls with a
    single fused Triton kernel that averages over 4x4x4 blocks where possible.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        # keep the original avg-pool modules for compatibility (not used in forward)
        self.avg_pool1 = nn.AvgPool3d(kernel_size=2)
        self.avg_pool2 = nn.AvgPool3d(kernel_size=2)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        # Use fused Triton kernel to replace two sequential avg pools
        x = triton_fused_avgpool3d(x)
        return x