import torch
import torch.nn as nn
import triton
import triton.language as tl

# Triton kernel that:
#  - expects input laid out as (N*D*H*W, C) contiguous, channels-last per-row
#  - computes logsumexp over the C dimension for each row
#  - applies hard-swish: y = lse * sigmoid(lse + 3) / 6
#  - subtracts a scalar bias and clamps to [-1, 1]
# Writes one float per row to out_ptr (shape N*D*H*W,)
@triton.jit
def _fused_lse_hswish_kernel(
    x_ptr,           # base pointer to input tensor (N, C, D, H, W)
    out_ptr,         # pointer to output floats (one per row)
    rows,            # number of rows = N*D*H*W
    C,               # number of channels
    N, D, H, W,      # dims (needed to compute coordinates)
    stride_n, stride_c, stride_d, stride_h, stride_w,  # strides in elements for each dim
    bias,            # scalar bias (float)
    BLOCK_ROWS: tl.constexpr,  # constexpr number of rows handled per program
    BLOCK_C: tl.constexpr,     # constexpr block size for channels
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_ROWS
    # vector of row indices this program will handle
    row_idxs = row_start + tl.arange(0, BLOCK_ROWS)
    mask_rows = row_idxs < rows

    # channel offsets
    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    # compute (n, d, h, w) for each row index
    HW = H * W
    DHW = D * HW

    n = row_idxs // DHW
    rem = row_idxs % DHW
    d = rem // HW
    rem2 = rem % HW
    h = rem2 // W
    w = rem2 % W

    # base offset (in elements) for each row: n*stride_n + d*stride_d + h*stride_h + w*stride_w
    base_offsets = n * stride_n + d * stride_d + h * stride_h + w * stride_w  # shape [BLOCK_ROWS]

    # column offsets (channel offsets) in elements
    col_offsets = offs_c * stride_c  # shape [BLOCK_C]

    # build pointer matrix of shape [BLOCK_ROWS, BLOCK_C]
    ptrs = x_ptr + base_offsets[:, None] + col_offsets[None, :]

    # load values with masking for both rows and channels
    vals = tl.load(ptrs, mask=(mask_rows[:, None] & mask_c[None, :]), other=-1e30)

    # compute max per row
    max_val = tl.max(vals, axis=1)  # shape [BLOCK_ROWS]

    # compute sum of exp(vals - max)
    exps = tl.exp(vals - max_val[:, None])
    exps = exps * mask_c[None, :].to(tl.float32)  # mask invalid channels
    sumexp = tl.sum(exps, axis=1)

    # log-sum-exp per row
    lse = max_val + tl.log(sumexp)

    # hard-swish: lse * sigmoid(lse + 3) / 6
    sig = 1.0 / (1.0 + tl.exp(-(lse + 3.0)))
    out = lse * sig / 6.0

    # subtract bias and clamp
    out = out - bias
    out = tl.maximum(out, -1.0)
    out = tl.minimum(out, 1.0)

    # write results for active rows
    tl.store(out_ptr + row_idxs, out, mask=mask_rows)


def fused_logsumexp_hardswish(x: torch.Tensor, bias: float):
    """
    x: tensor of shape (N, C, D, H, W)
    bias: scalar float to subtract
    returns tensor of shape (N, 1, D, H, W) (same device and dtype as x)
    """
    assert x.is_cuda, "Input must be on CUDA."
    assert x.dim() == 5, "Expected input of shape (N, C, D, H, W)."

    N, C, D, H, W = x.shape

    rows = N * D * H * W

    # Prepare output flat
    out_flat = torch.empty((rows,), dtype=x.dtype, device=x.device)

    # get strides (in elements) of the tensor; we use these to compute element addresses inside the kernel
    sN, sC, sD, sH, sW = x.stride()

    # Choose tiling parameters: BLOCK_ROWS and BLOCK_C
    # Increase BLOCK_ROWS to improve coalescing; set BLOCK_C exactly to C when small to avoid padding
    BLOCK_ROWS = 64
    # set BLOCK_C to C for small channel counts (<=32) to avoid unnecessary padding.
    if C <= 32:
        BLOCK_C = C
    else:
        # round up to nearest multiple of 32 for larger C to satisfy vectorization widths
        BLOCK_C = ((C + 31) // 32) * 32

    # number of program instances required
    grid = ((rows + BLOCK_ROWS - 1) // BLOCK_ROWS,)

    # Launch kernel; pass the tensor (Triton will use its pointer) and strides
    _fused_lse_hswish_kernel[grid](
        x, out_flat, rows, C, N, D, H, W,
        sN, sC, sD, sH, sW,
        float(bias),
        BLOCK_ROWS=BLOCK_ROWS, BLOCK_C=BLOCK_C
    )

    # Reshape back to (N, 1, D, H, W)
    out = out_flat.view(N, D, H, W).unsqueeze(1)  # (N,1,D,H,W)

    return out


class ModelNew(nn.Module):
    """
    Optimized model:
      - Uses the original ConvTranspose3d for the heavy convolution transpose.
      - Replaces the subsequent channel-wise logsumexp, hard-swish, subtraction, and clamp
        with a fused Triton kernel for improved throughput.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        # keep bias as a parameter to match original model; the fused kernel accepts a scalar bias value
        # Note: Triton kernel currently is not autograd-tracked, so gradients through this fused op
        # will not flow back. This implementation targets inference performance.
        self.bias = nn.Parameter(torch.randn(*bias_shape))

    def forward(self, x):
        # x: (N, in_channels, D, H, W)
        # conv_transpose -> (N, out_channels, D', H', W')
        x = self.conv_transpose(x)

        # fused logsumexp over channels + hard-swish + subtract bias + clamp
        # bias is broadcastable; we take its scalar value for subtraction
        bias_scalar = float(self.bias.reshape(-1)[0].item())

        out = fused_logsumexp_hardswish(x, bias_scalar)
        return out


# Keep get_inputs and get_init_inputs similar to the original specification
batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
bias_shape = (1, 1, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda().float()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias_shape]