import torch
import torch.nn as nn
import triton
import triton.language as tl

# Fused Triton kernels: compute per-group statistics with atomics (partial reductions)
# and apply fused bias, scale, sigmoid, and GroupNorm affine.
# Tuned block sizes chosen heuristically based on spatial size to improve throughput
# on A6000 (Ampere). These choices aim to reduce kernel-launch overhead and atomic
# contention while keeping good occupancy.

@triton.jit
def _compute_stats_kernel(
    x_ptr,            # *ptr to input tensor (N, C, H, W) flattened
    bias_ptr,         # per-channel bias (C,)
    scale_ptr,        # per-channel scale (C,)
    sums_ptr,         # output sums per (N, G)
    sumsq_ptr,        # output sumsq per (N, G)
    N, C, H, W,       # shapes
    group_size,       # channels per group
    S,                # spatial size = H*W
    elems_f,          # float(total_elems) where total_elems = group_size * S
    BLOCK: tl.constexpr
):
    # program ids: (n, g, tile)
    n = tl.program_id(0)
    g = tl.program_id(1)
    tile = tl.program_id(2)
    c_base = g * group_size
    # spatial tile start for this program
    start_sp = tile * BLOCK
    offs = tl.arange(0, BLOCK)
    idx = start_sp + offs
    mask = idx < S

    # accumulate scalar sums for this tile across channels in the group
    acc = 0.0
    acc2 = 0.0
    ch = 0
    # Loop over channels within the group
    while ch < group_size:
        # per-channel pointer base for (n, c)
        ptr_ch_base = x_ptr + n * (C * S) + (c_base + ch) * S
        # load per-channel bias/scale
        b = tl.load(bias_ptr + c_base + ch)
        sc = tl.load(scale_ptr + c_base + ch)
        # load this tile of spatial values for this channel
        ptr = ptr_ch_base + idx
        vals = tl.load(ptr, mask=mask, other=0.0)
        # apply bias, scale, sigmoid
        y = (vals + b) * sc
        y = 1.0 / (1.0 + tl.exp(-y))
        # mask out-of-bounds contributions
        masked_y = tl.where(mask, y, 0.0)
        masked_y2 = tl.where(mask, y * y, 0.0)
        acc += tl.sum(masked_y)
        acc2 += tl.sum(masked_y2)
        ch += 1

    # atomic add partial results into global sums for this (n,g)
    out_index = n * (C // group_size) + g
    tl.atomic_add(sums_ptr + out_index, acc)
    tl.atomic_add(sumsq_ptr + out_index, acc2)


@triton.jit
def _apply_pointwise_groupnorm_kernel(
    x_ptr,             # input tensor ptr (N,C,H,W)
    out_ptr,           # output tensor ptr (N,C,H,W)
    bias_ptr,          # per-channel bias (C,)
    scale_ptr,         # per-channel scale (C,)
    gn_weight_ptr,     # groupnorm weight per channel (C,)
    gn_bias_ptr,       # groupnorm bias per channel (C,)
    sums_ptr,          # sums per (N, G)
    sumsq_ptr,         # sumsq per (N, G)
    N, C, H, W,
    group_size,
    S,
    elems_f,
    eps: tl.constexpr,
    BLOCK: tl.constexpr
):
    # program ids: (n, c, tile)
    n = tl.program_id(0)
    c = tl.program_id(1)
    tile = tl.program_id(2)
    g = c // group_size
    out_index = n * (C // group_size) + g

    # load mean and var for this (n,g) once
    s = tl.load(sums_ptr + out_index)
    ss = tl.load(sumsq_ptr + out_index)
    mean = s / elems_f
    var = ss / elems_f - mean * mean
    invstd = 1.0 / tl.sqrt(var + eps)

    # load per-channel params once
    b_c = tl.load(bias_ptr + c)
    sc_c = tl.load(scale_ptr + c)
    gw = tl.load(gn_weight_ptr + c)
    gb = tl.load(gn_bias_ptr + c)

    # spatial tile for this program
    start = tile * BLOCK
    offs = tl.arange(0, BLOCK)
    idx = start + offs
    mask = idx < S
    ptr = x_ptr + n * (C * S) + c * S + idx
    vals = tl.load(ptr, mask=mask, other=0.0)
    # apply bias, scale, sigmoid
    y = (vals + b_c) * sc_c
    y = 1.0 / (1.0 + tl.exp(-y))
    # normalize and apply groupnorm affine
    y = (y - mean) * invstd
    out = y * gw + gb
    tl.store(out_ptr + n * (C * S) + c * S + idx, out, mask=mask)


class ModelNew(nn.Module):
    """
    Optimized Model that keeps PyTorch's Conv2d for convolution and uses Triton
    to fuse bias, scale, sigmoid, and GroupNorm. This implementation heuristically
    selects block sizes based on spatial size to reduce kernel-launch overhead
    and atomic contention on Ampere GPUs (A6000).
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super(ModelNew, self).__init__()
        # Use PyTorch Conv2d (cuDNN) for convolution performance
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # per-channel bias and scale used before sigmoid
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        # GroupNorm retains affine params and num_groups
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        # x: (N, in_channels, H, W)
        x = self.conv(x)  # (N, C, H, W)
        assert x.is_cuda, "This optimized model requires CUDA tensors on CUDA device."
        N, C, H, W = x.shape
        x = x.contiguous()

        # flatten per-channel parameters
        bias_flat = self.bias.view(C).contiguous()
        scale_flat = self.scale.view(C).contiguous()

        # GroupNorm affine params (fall back to ones/zeros if None)
        if self.group_norm.weight is None:
            gn_w = torch.ones(C, device=x.device, dtype=x.dtype)
        else:
            gn_w = self.group_norm.weight.view(C).contiguous()
        if self.group_norm.bias is None:
            gn_b = torch.zeros(C, device=x.device, dtype=x.dtype)
        else:
            gn_b = self.group_norm.bias.view(C).contiguous()

        num_groups = self.group_norm.num_groups
        group_size = C // num_groups
        S = H * W
        elems = group_size * S
        elems_f = float(elems)

        # allocate sums and sumsq per (N, G) and zero them before atomic adds
        stats_shape = (N * num_groups,)
        sums = torch.empty(stats_shape, dtype=x.dtype, device=x.device)
        sumsq = torch.empty(stats_shape, dtype=x.dtype, device=x.device)
        sums.zero_()
        sumsq.zero_()

        # Heuristic selection of BLOCK sizes tuned for large spatial sizes on A6000.
        # We clamp to powers of two that are reasonable for Triton register usage.
        if S >= 65536:
            BLOCK_STATS = 4096
            BLOCK_APPLY = 4096
        elif S >= 32768:
            BLOCK_STATS = 2048
            BLOCK_APPLY = 2048
        elif S >= 16384:
            BLOCK_STATS = 2048
            BLOCK_APPLY = 1024
        else:
            BLOCK_STATS = 1024
            BLOCK_APPLY = 512

        # Launch compute stats kernel: grid (N, num_groups, num_tiles)
        num_tiles_stats = (S + BLOCK_STATS - 1) // BLOCK_STATS
        grid_stats = (N, num_groups, num_tiles_stats)
        _compute_stats_kernel[grid_stats](
            x,                     # x_ptr
            bias_flat,             # bias per channel
            scale_flat,            # scale per channel
            sums,                  # output sums per (N,G)
            sumsq,                 # output sumsq per (N,G)
            N, C, H, W,
            group_size,
            S,
            elems_f,
            BLOCK=BLOCK_STATS
        )

        # allocate output
        out = torch.empty_like(x)

        # Launch apply kernel: grid (N, C, num_tiles_apply)
        num_tiles_apply = (S + BLOCK_APPLY - 1) // BLOCK_APPLY
        grid_apply = (N, C, num_tiles_apply)
        eps = 1e-5
        _apply_pointwise_groupnorm_kernel[grid_apply](
            x,
            out,
            bias_flat,
            scale_flat,
            gn_w,
            gn_b,
            sums,
            sumsq,
            N, C, H, W,
            group_size,
            S,
            elems_f,
            eps,
            BLOCK=BLOCK_APPLY
        )

        return out