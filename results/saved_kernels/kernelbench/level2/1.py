import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configurations tuned for NVIDIA A6000 (Ampere).
# Explore a range of BLOCK sizes and ROWS_PER_PROG to find the best throughput.
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 128,  "ROWS_PER_PROG": 1}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 128,  "ROWS_PER_PROG": 2}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 128,  "ROWS_PER_PROG": 4}, num_warps=8, num_stages=2),

    triton.Config({"BLOCK": 256,  "ROWS_PER_PROG": 1}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 256,  "ROWS_PER_PROG": 2}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK": 256,  "ROWS_PER_PROG": 4}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK": 256,  "ROWS_PER_PROG": 8}, num_warps=8, num_stages=3),

    triton.Config({"BLOCK": 512,  "ROWS_PER_PROG": 1}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK": 512,  "ROWS_PER_PROG": 2}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK": 512,  "ROWS_PER_PROG": 4}, num_warps=8, num_stages=3),

    # try a larger BLOCK to increase memory throughput on Ampere
    triton.Config({"BLOCK": 1024, "ROWS_PER_PROG": 1}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK": 1024, "ROWS_PER_PROG": 2}, num_warps=8, num_stages=3),
]

@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['n_rows', 'W', 'H', 'C'])
@triton.jit
def _fused_relu_add_bias_rowwise(
    x_ptr,         # pointer to input tensor (N*C*H*W), contiguous
    bias_ptr,      # pointer to bias (C,)
    out_ptr,       # pointer to output tensor
    n_rows,        # N * C * H (number of rows)
    W,             # width (elements per row)
    H,             # height (used to recover channel index)
    C,             # channels
    BLOCK: tl.constexpr,         # number of elements processed per program along W
    ROWS_PER_PROG: tl.constexpr, # number of rows handled per program
):
    """
    Each program processes ROWS_PER_PROG consecutive rows and a contiguous block
    of BLOCK elements along the width dimension. This layout favors memory
    coalescing and reuses the scalar bias per-channel for all vector lanes.
    """
    group = tl.program_id(0)   # which group of rows (each group contains ROWS_PER_PROG rows)
    block_id = tl.program_id(1)
    col_start = block_id * BLOCK
    offs = col_start + tl.arange(0, BLOCK)  # offsets within a row
    mask_cols = offs < W                      # mask for valid columns

    base_row = group * ROWS_PER_PROG

    # Unroll across the small constexpr ROWS_PER_PROG
    for i in range(ROWS_PER_PROG):
        row = base_row + i
        within_rows = row < n_rows  # boolean scalar representing validity of the row

        # compute flattened offsets for this row: row * W + offs
        row_start = row * W
        flat_offs = row_start + offs

        # combine masks: both column valid and row valid
        mask = mask_cols & within_rows

        # Load inputs for this segment (masked)
        x_vals = tl.load(x_ptr + flat_offs, mask=mask, other=0.0)

        # recover channel index:
        # row = ((n * C) + c) * H + h  => tmp = row // H = n*C + c  => c = tmp % C
        tmp = row // H
        c = tmp % C

        # Load bias scalar for this channel and broadcast-add
        bias_val = tl.load(bias_ptr + c)

        # fused op: relu then add bias
        relu = tl.where(x_vals > 0.0, x_vals, 0.0)
        out_vals = relu + bias_val

        # Store results back (masked)
        tl.store(out_ptr + flat_offs, out_vals, mask=mask)


def triton_fused_relu_add_bias(x: torch.Tensor, bias: torch.Tensor):
    """
    Wrapper around the Triton kernel.
    x: (N, C, H, W) contiguous CUDA tensor (fp32)
    bias: (C,) or (C,1,1) tensor; will be flattened
    Returns a new tensor with relu applied and per-channel bias added.
    """
    assert x.is_cuda and bias.is_cuda, "Tensors must be on CUDA to run Triton kernel."

    # Ensure contiguous layout for pointer arithmetic in Triton kernel
    x = x.contiguous()
    bias_flat = bias.view(-1).contiguous()

    N, C, H, W = x.shape
    n_rows = N * C * H
    out = torch.empty_like(x)

    # Grid is 2D: number of groups (over rows) x number of blocks per row
    grid = lambda meta: (
        (n_rows + meta['ROWS_PER_PROG'] - 1) // meta['ROWS_PER_PROG'],
        (W + meta['BLOCK'] - 1) // meta['BLOCK'],
    )

    # Launch kernel
    _fused_relu_add_bias_rowwise[grid](x, bias_flat, out, n_rows, W, H, C)
    return out


class ModelNew(nn.Module):
    """
    Optimized Model:
      - Uses PyTorch's highly-optimized Conv2d for convolution.
      - Fuses ReLU + per-channel bias add into a Triton kernel that operates
        row-wise (along width) and is autotuned for Ampere GPUs.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # Keep the additional bias (as in the original model)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv(x)
        # If input is on CUDA, use the fused Triton kernel for ReLU + bias add
        if x.is_cuda:
            return triton_fused_relu_add_bias(x, self.bias)
        else:
            # CPU fallback: do the operations in PyTorch
            x = torch.relu(x)
            return x + self.bias