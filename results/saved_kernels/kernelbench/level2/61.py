import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configurations to let Triton pick the best TILE and number of warps/stages
AUTOTUNE_CONFIGS = [
    triton.Config({"TILE": 1024}, num_warps=4, num_stages=2),
    triton.Config({"TILE": 2048}, num_warps=4, num_stages=2),
    triton.Config({"TILE": 4096}, num_warps=8, num_stages=2),
    triton.Config({"TILE": 8192}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['M'])
@triton.jit
def _group_sum_kernel(
    x_ptr,            # input tensor pointer flattened
    sums_ptr,         # output pointer for sums per (N*G)
    sumsqs_ptr,       # output pointer for sumsq per (N*G)
    N,                # batch size
    C,                # total channels
    G,                # number of groups
    Cg,               # channels per group
    S,                # spatial size (D*H*W)
    M,                # Cg * S
    TILE: tl.constexpr,  # tile size (number of elements processed per iteration)
):
    """
    One program per (n, g) row. Each program loops over the group's M elements in chunks of TILE.
    Reads input, applies ReLU, accumulates sum and sumsq.
    """
    row = tl.program_id(0)
    if row >= N * G:
        return

    n = row // G
    g = row % G

    base = ((n * C + g * Cg) * S)  # base offset in flattened elements for this group

    acc = 0.0
    accsq = 0.0

    # iterate tiles
    offs = tl.arange(0, TILE)
    tile_start = 0
    while tile_start < M:
        idx = base + tile_start + offs
        mask = idx < (base + M)
        vals = tl.load(x_ptr + idx, mask=mask, other=0.0)
        # fused ReLU
        vals = tl.where(vals > 0.0, vals, 0.0)
        acc += tl.sum(vals)
        accsq += tl.sum(vals * vals)
        tile_start += TILE

    tl.store(sums_ptr + row, acc)
    tl.store(sumsqs_ptr + row, accsq)


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['M'])
@triton.jit
def _group_norm_kernel(
    x_ptr,            # input tensor pointer flattened
    out_ptr,          # output tensor pointer flattened
    sums_ptr,         # pointer for sums per (N*G)
    sumsqs_ptr,       # pointer for sumsq per (N*G)
    N,                # batch size
    C,                # total channels
    G,                # number of groups
    Cg,               # channels per group
    S,                # spatial size (D*H*W)
    M,                # Cg * S
    eps,
    TILE: tl.constexpr,  # tile size
):
    """
    One program per (n, g) row. Reads precomputed sums to compute mean/var/invstd,
    then loops over group's elements in tiles, applies ReLU, normalizes, and writes output.
    """
    row = tl.program_id(0)
    if row >= N * G:
        return

    n = row // G
    g = row % G
    base = ((n * C + g * Cg) * S)

    s = tl.load(sums_ptr + row)
    ss = tl.load(sumsqs_ptr + row)

    mean = s / M
    var = ss / M - mean * mean
    invstd = 1.0 / tl.sqrt(var + eps)

    offs = tl.arange(0, TILE)
    tile_start = 0
    while tile_start < M:
        idx = base + tile_start + offs
        mask = idx < (base + M)
        vals = tl.load(x_ptr + idx, mask=mask, other=0.0)
        vals = tl.where(vals > 0.0, vals, 0.0)  # ReLU
        out_vals = (vals - mean) * invstd
        tl.store(out_ptr + idx, out_vals, mask=mask)
        tile_start += TILE


def triton_groupnorm_relu(x: torch.Tensor, num_groups: int, eps: float = 1e-5):
    """
    Apply ReLU followed by GroupNorm using optimized Triton kernels.
    This function:
      - Flattens the input to (N, C, S) contiguous layout
      - Runs a reduction kernel to compute per-(N,G) sums and sumsqs (with ReLU applied on the fly)
      - Runs a normalization kernel that applies ReLU again and writes normalized outputs
    """
    assert x.is_cuda, "Input must be a CUDA tensor."
    x = x.contiguous()
    N, C, D, H, W = x.shape
    assert C % num_groups == 0, "num_groups must divide channels"
    G = num_groups
    Cg = C // G
    S = D * H * W
    M = Cg * S

    out = torch.empty_like(x)

    sums = torch.empty((N * G,), dtype=torch.float32, device=x.device)
    sumsqs = torch.empty((N * G,), dtype=torch.float32, device=x.device)

    # Launch reduction kernel: one program per (N*G) row
    grid = (N * G,)
    _group_sum_kernel[grid](
        x,
        sums,
        sumsqs,
        N,
        C,
        G,
        Cg,
        S,
        M,
    )

    # Launch normalization kernel
    _group_norm_kernel[grid](
        x,
        out,
        sums,
        sumsqs,
        N,
        C,
        G,
        Cg,
        S,
        M,
        float(eps),
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized model that keeps ConvTranspose3d implemented by PyTorch (leveraging highly-tuned
    cuDNN/cuBLAS code paths) but replaces the ReLU + GroupNorm sequence with a fused
    Triton implementation that computes ReLU and GroupNorm efficiently in two fused kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super(ModelNew, self).__init__()
        # Keep convolution transpose in PyTorch for correctness and performance
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, bias=bias)
        self.groups = groups
        self.eps = 1e-5

    def forward(self, x):
        x = self.conv_transpose(x)
        x = triton_groupnorm_relu(x, self.groups, eps=self.eps)
        return x