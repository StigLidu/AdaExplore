import torch
import torch.nn as nn
import triton
import triton.language as tl

# Fused Triton kernel: softmax over channels, add per-channel bias, scale, and sigmoid.
# Each Triton program handles a small tile of shape (ROWS_PER_PROG, BLOCK) where
# ROWS_PER_PROG is number of spatial positions (n,h,w combos) and BLOCK is number of channels.
@triton.jit
def _fused_softmax_bias_scale_sigmoid_kernel(
    x_ptr,            # base pointer for input tensor in NCHW layout (fp32)
    out_ptr,          # base pointer for output tensor in NCHW layout (fp32)
    bias_ptr,         # bias pointer, shape (C,)
    N,                # batch dim
    C,                # number of channels
    H,                # height
    W,                # width
    stride_n,         # stride for N dimension (in elements)
    stride_c,         # stride for C dimension (in elements)
    stride_h,         # stride for H dimension (in elements)
    stride_w,         # stride for W dimension (in elements)
    scaling,          # scalar scaling factor (float32)
    ROWS_PER_PROG: tl.constexpr,  # how many rows (spatial positions) each program handles
    BLOCK: tl.constexpr           # how many channels each program handles (must be constexpr)
):
    # program id and the starting row index
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_PROG
    row_offsets = row_start + tl.arange(0, ROWS_PER_PROG)  # shape (ROWS_PER_PROG,)

    HW = H * W
    # Compute n, h, w for each row offset
    n = row_offsets // HW
    hw = row_offsets % HW
    h = hw // W
    w = hw % W

    c_offs = tl.arange(0, BLOCK)  # shape (BLOCK,)

    # mask for valid rows and valid channels
    valid_row = row_offsets < (N * HW)
    mask = valid_row[:, None] & (c_offs[None, :] < C)  # shape (ROWS_PER_PROG, BLOCK)

    # addresses into the NCHW layout for each (row, channel)
    # addr[r, c] = n[r]*stride_n + c*stride_c + h[r]*stride_h + w[r]*stride_w
    addr = n[:, None] * stride_n + c_offs[None, :] * stride_c + h[:, None] * stride_h + w[:, None] * stride_w

    # Load input values. Use a large negative other so masked lanes don't affect max.
    vals = tl.load(x_ptr + addr, mask=mask, other=-1e20)  # shape (ROWS_PER_PROG, BLOCK)

    # Numerically stable softmax across channels per row
    m = tl.max(vals, axis=1)                # shape (ROWS_PER_PROG,)
    vals = vals - m[:, None]
    exp_vals = tl.exp(vals)
    denom = tl.sum(exp_vals, axis=1)        # shape (ROWS_PER_PROG,)
    softmax = exp_vals / denom[:, None]     # shape (ROWS_PER_PROG, BLOCK)

    # load bias (1D over channels) and broadcast to rows
    bias_vals = tl.load(bias_ptr + c_offs, mask=c_offs < C, other=0.0)  # shape (BLOCK,)
    bias_vals_b = bias_vals[None, :]  # broadcast over rows

    # add bias, scale, and apply sigmoid
    out_vals = softmax + bias_vals_b
    out_vals = out_vals * scaling
    out_vals = 1.0 / (1.0 + tl.exp(-out_vals))

    # store results back to original NCHW layout
    tl.store(out_ptr + addr, out_vals, mask=mask)


def fused_softmax_bias_scale_sigmoid(x: torch.Tensor, bias: torch.Tensor, scaling: float):
    """
    Fused operator: softmax over channel dim (dim=1) of x (N,C,H,W),
    add per-channel bias, multiply by scaling, then apply sigmoid.
    Uses Triton on CUDA and falls back to pure PyTorch on CPU.
    """
    if not x.is_cuda:
        out = torch.softmax(x, dim=1)
        out = out + bias
        out = out * scaling
        out = torch.sigmoid(out)
        return out

    # Ensure expected dtype
    assert x.dtype == torch.float32, "Expected float32 inputs."

    N, C, H, W = x.shape
    rows = N * H * W
    out = torch.empty_like(x)

    # Flatten bias to shape (C,)
    bias_flat = bias.contiguous().view(-1).to(x.device)

    # Strides (in elements)
    stride_n, stride_c, stride_h, stride_w = x.stride()

    # Heuristic tuning for BLOCK and ROWS_PER_PROG to adapt to shapes / GPU
    # These heuristics aim to balance parallelism across channels and spatial positions.
    if C >= 512:
        BLOCK = 512
    elif C >= 256:
        BLOCK = 256
    else:
        # keep BLOCK a multiple of 32 for warp efficiency
        BLOCK = 128 if C >= 128 else 32

    # Choose rows per program based on total rows to balance launch overhead
    if rows >= 16384:
        ROWS_PER_PROG = 16
    elif rows >= 4096:
        ROWS_PER_PROG = 8
    else:
        ROWS_PER_PROG = 4

    # Ensure BLOCK does not exceed C too much (keeps kernel occupancy reasonable)
    # If BLOCK > C, kernel masking will handle it; but reduce BLOCK to avoid wasted resources.
    if BLOCK > C:
        # pick the largest multiple of 32 <= C
        BLOCK = max(32, (C // 32) * 32)

    # grid: number of Triton programs needed to cover all rows
    num_programs = (rows + ROWS_PER_PROG - 1) // ROWS_PER_PROG
    grid = lambda meta: (num_programs,)

    # Launch kernel
    _fused_softmax_bias_scale_sigmoid_kernel[grid](
        x, out, bias_flat, N, C, H, W,
        stride_n, stride_c, stride_h, stride_w,
        float(scaling),
        ROWS_PER_PROG=ROWS_PER_PROG, BLOCK=BLOCK
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized Model that performs a ConvTranspose2d followed by a fused Triton kernel:
    softmax over channels, add per-channel bias, scale, and sigmoid.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        # Keep bias as learnable parameter, shaped (C,1,1) like original model
        self.bias = nn.Parameter(torch.randn(bias_shape, dtype=torch.float32))
        self.scaling_factor = float(scaling_factor)

    def forward(self, x):
        x = self.conv_transpose(x)
        # If on CUDA, use Triton fused kernel for the channel-wise operations.
        # If on CPU, fall back to PyTorch operations.
        return fused_softmax_bias_scale_sigmoid(x, self.bias, self.scaling_factor)