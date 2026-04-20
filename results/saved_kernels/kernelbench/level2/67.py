import torch
import torch.nn as nn
import triton
import triton.language as tl

# Expanded autotune search exploring larger ROWS to amortize launch cost and larger BLOCK to maximize bandwidth.
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 256, "ROWS": 2},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 512, "ROWS": 2},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 1024, "ROWS": 2}, num_warps=8, num_stages=2),

    triton.Config({"BLOCK": 512, "ROWS": 4},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 1024, "ROWS": 4}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK": 2048, "ROWS": 4}, num_warps=8, num_stages=3),

    triton.Config({"BLOCK": 1024, "ROWS": 8}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK": 2048, "ROWS": 8}, num_warps=8, num_stages=3),
]

@triton.autotune(AUTOTUNE_CONFIGS, key=['ncols', 'nrows'])
@triton.jit
def _fused_gelu_avgpool_kernel(x_ptr, out_ptr, ncols, nrows, BLOCK: tl.constexpr, ROWS: tl.constexpr):
    """
    Fused GELU (sigmoid approximation) + global average pooling across spatial dims.
    Each Triton program handles 'ROWS' output rows (each row corresponds to one (batch,channel) pair).
    We iterate across columns (flattened H*W) in BLOCK-sized chunks with an unroll factor to maximize bandwidth.
    """
    pid = tl.program_id(0)
    row_start = pid * ROWS

    offs = tl.arange(0, BLOCK)
    UNROLL = 4  # Unroll factor to increase sustained bandwidth while keeping register pressure reasonable.

    inv_ncols = 1.0 / ncols

    # Initialize accumulators for up to 8 rows (support ROWS up to 8 in autotune configs)
    acc0 = 0.0
    acc1 = 0.0
    acc2 = 0.0
    acc3 = 0.0
    acc4 = 0.0
    acc5 = 0.0
    acc6 = 0.0
    acc7 = 0.0

    # Loop across the flattened spatial dimension in chunks
    for start in range(0, ncols, BLOCK * UNROLL):
        # Unrolled loads to maximize contiguous memory throughput
        for u in range(UNROLL):
            idx = start + u * BLOCK + offs
            mask = idx < ncols

            # Row 0
            r = row_start + 0
            if r < nrows:
                ptr = x_ptr + r * ncols + idx
                vals = tl.load(ptr, mask=mask, other=0.0)
                sig = 1.0 / (1.0 + tl.exp(-1.702 * vals))
                g = vals * sig
                acc0 += tl.sum(g)

            # Row 1
            r = row_start + 1
            if ROWS > 1 and r < nrows:
                ptr = x_ptr + r * ncols + idx
                vals = tl.load(ptr, mask=mask, other=0.0)
                sig = 1.0 / (1.0 + tl.exp(-1.702 * vals))
                g = vals * sig
                acc1 += tl.sum(g)

            # Row 2
            r = row_start + 2
            if ROWS > 2 and r < nrows:
                ptr = x_ptr + r * ncols + idx
                vals = tl.load(ptr, mask=mask, other=0.0)
                sig = 1.0 / (1.0 + tl.exp(-1.702 * vals))
                g = vals * sig
                acc2 += tl.sum(g)

            # Row 3
            r = row_start + 3
            if ROWS > 3 and r < nrows:
                ptr = x_ptr + r * ncols + idx
                vals = tl.load(ptr, mask=mask, other=0.0)
                sig = 1.0 / (1.0 + tl.exp(-1.702 * vals))
                g = vals * sig
                acc3 += tl.sum(g)

            # Row 4
            r = row_start + 4
            if ROWS > 4 and r < nrows:
                ptr = x_ptr + r * ncols + idx
                vals = tl.load(ptr, mask=mask, other=0.0)
                sig = 1.0 / (1.0 + tl.exp(-1.702 * vals))
                g = vals * sig
                acc4 += tl.sum(g)

            # Row 5
            r = row_start + 5
            if ROWS > 5 and r < nrows:
                ptr = x_ptr + r * ncols + idx
                vals = tl.load(ptr, mask=mask, other=0.0)
                sig = 1.0 / (1.0 + tl.exp(-1.702 * vals))
                g = vals * sig
                acc5 += tl.sum(g)

            # Row 6
            r = row_start + 6
            if ROWS > 6 and r < nrows:
                ptr = x_ptr + r * ncols + idx
                vals = tl.load(ptr, mask=mask, other=0.0)
                sig = 1.0 / (1.0 + tl.exp(-1.702 * vals))
                g = vals * sig
                acc6 += tl.sum(g)

            # Row 7
            r = row_start + 7
            if ROWS > 7 and r < nrows:
                ptr = x_ptr + r * ncols + idx
                vals = tl.load(ptr, mask=mask, other=0.0)
                sig = 1.0 / (1.0 + tl.exp(-1.702 * vals))
                g = vals * sig
                acc7 += tl.sum(g)

    # Store the averaged results back to global memory
    r = row_start + 0
    if r < nrows:
        tl.store(out_ptr + r, acc0 * inv_ncols)
    r = row_start + 1
    if ROWS > 1 and r < nrows:
        tl.store(out_ptr + r, acc1 * inv_ncols)
    r = row_start + 2
    if ROWS > 2 and r < nrows:
        tl.store(out_ptr + r, acc2 * inv_ncols)
    r = row_start + 3
    if ROWS > 3 and r < nrows:
        tl.store(out_ptr + r, acc3 * inv_ncols)
    r = row_start + 4
    if ROWS > 4 and r < nrows:
        tl.store(out_ptr + r, acc4 * inv_ncols)
    r = row_start + 5
    if ROWS > 5 and r < nrows:
        tl.store(out_ptr + r, acc5 * inv_ncols)
    r = row_start + 6
    if ROWS > 6 and r < nrows:
        tl.store(out_ptr + r, acc6 * inv_ncols)
    r = row_start + 7
    if ROWS > 7 and r < nrows:
        tl.store(out_ptr + r, acc7 * inv_ncols)


def triton_gelu_avgpool(x: torch.Tensor) -> torch.Tensor:
    """
    Fused post-processing:
      - Applies GELU (fast sigmoid-based approximation) and global average pooling across H*W,
        returning a tensor shaped (batch, channels).
    """
    assert x.is_cuda, "triton_gelu_avgpool expects CUDA tensors"
    x = x.contiguous()
    batch, channels, height, width = x.shape
    nrows = batch * channels
    ncols = height * width

    # Flatten spatial dims
    x_flat = x.view(nrows, ncols)

    out = torch.empty(nrows, device=x.device, dtype=x.dtype)

    # grid: one Triton program handles ROWS output rows; meta supplies BLOCK and ROWS
    grid = lambda meta: ((nrows + meta["ROWS"] - 1) // meta["ROWS"],)

    # Launch the autotuned kernel (meta selected by triton.autotune)
    _fused_gelu_avgpool_kernel[grid](x_flat, out, ncols, nrows)
    return out.view(batch, channels)


class ModelNew(nn.Module):
    """
    Optimized model:
      - Keep PyTorch's highly-optimized Conv2d
      - Fuse GELU + global average pool into a single Triton kernel for reduced memory traffic and latency.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        # Convolution uses PyTorch/CuDNN path (user/harness should move model to device as needed)
        x = self.conv(x)
        # Fused GELU + global average pooling via Triton
        x = triton_gelu_avgpool(x)
        return x


# Keep helper variables and input functions compatible with the harness
batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3

def get_inputs():
    # Return CUDA tensors for the harness; Triton kernel requires CUDA inputs.
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]