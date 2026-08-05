import torch
import torch.nn as nn
import triton
import triton.language as tl

# Triton kernels implementing fused ReLU + GroupNorm (affine).
# Strategy (revised):
#  1) Tiled partial reduction: each program handles one tile of a group's elements and emits a partial sum/sumsq.
#  2) Final reduction: sum the partials per (N,G) on device (torch.sum over tile dim).
#  3) Chunked normalization: grid over (N*C*n_chunks,) so each program handles one TILE-sized spatial chunk
#     for a fixed (n,c), applies ReLU, normalizes with precomputed mean/invstd and applies affine transform.

@triton.jit
def _group_stats_partial_kernel(
    x_ptr,                 # pointer to input tensor (N, C, S) flattened
    partial_sum_ptr,       # pointer to partial sums (N*G*P,)
    partial_sumsq_ptr,     # pointer to partial sumsqs (N*G*P,)
    N, C, S, G, group_elems, P,
    BLOCK: tl.constexpr
):
    # grid launched with (N*G*P,) programs
    pid = tl.program_id(0)                     # pid in [0, N*G*P)
    p = pid // (N * G)                         # partition index in [0, P)
    rem = pid % (N * G)
    n = rem // G
    g = rem % G

    group_size = C // G
    c0 = g * group_size
    # start offset of this group's block in flattened x (layout: n, c, s)
    start = (n * C + c0) * S

    # partition size (last partition may be smaller)
    part_size = (group_elems + P - 1) // P

    base = start + p * part_size

    offs = tl.arange(0, BLOCK)

    # accumulate over the whole partition in BLOCK-sized strides
    acc = 0.0
    acc2 = 0.0
    n_iters = (part_size + BLOCK - 1) // BLOCK
    for i in range(n_iters):
        cur_offs = base + i * BLOCK + offs
        # mask out-of-range elements (either beyond group's end or beyond this partition)
        mask1 = cur_offs < (start + group_elems)
        mask2 = cur_offs < (base + part_size)
        mask = mask1 & mask2
        vals = tl.load(x_ptr + cur_offs, mask=mask, other=0.0)
        vals = tl.maximum(vals, 0.0)  # ReLU
        acc += tl.sum(vals)
        acc2 += tl.sum(vals * vals)

    # store partials at index: rem * P + p
    idx = rem * P + p
    tl.store(partial_sum_ptr + idx, acc)
    tl.store(partial_sumsq_ptr + idx, acc2)


@triton.jit
def _groupnorm_affine_kernel_chunked(
    x_ptr, out_ptr,
    mean_ptr, invstd_ptr,
    weight_ptr, bias_ptr,
    N, C, S, G, group_size, n_chunks,
    BLOCK: tl.constexpr
):
    # grid launched with (N*C*n_chunks,) programs, each program handles a fixed (n,c,chunk)
    pid = tl.program_id(0)
    chunk = pid % n_chunks
    pair = pid // n_chunks
    n = pair // C
    c = pair % C

    # precompute per-program constants
    channel_start = (n * C + c) * S
    base = channel_start + chunk * BLOCK

    offs = tl.arange(0, BLOCK)
    idx = base + offs
    mask = idx < (channel_start + S)

    vals = tl.load(x_ptr + idx, mask=mask, other=0.0)
    vals = tl.maximum(vals, 0.0)  # ReLU

    group = c // group_size
    mean_idx = n * G + group

    # load scalars once per program
    mean = tl.load(mean_ptr + mean_idx)
    invstd = tl.load(invstd_ptr + mean_idx)
    w = tl.load(weight_ptr + c)
    b = tl.load(bias_ptr + c)

    normalized = (vals - mean) * invstd
    out = normalized * w + b
    tl.store(out_ptr + idx, out, mask=mask)


def _fused_relu_groupnorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, groups: int, eps: float = 1e-5):
    assert x.is_cuda and weight.is_cuda and (bias is None or bias.is_cuda), "All tensors must be on CUDA."
    assert x.dtype == torch.float32 and weight.dtype == torch.float32 and (bias is None or bias.dtype == torch.float32)

    # x shape: (N, C, D, H, W)
    N, C, D, H, W = x.shape
    S = D * H * W
    G = groups
    assert C % G == 0, "channels must be divisible by groups"
    group_size = C // G
    group_elems = group_size * S

    x_contig = x.contiguous()

    # Tiling parameter: TILE (BLOCK) chosen for Ampere (multiple of 32)
    BLOCK = 256

    # --- Tiled partial reduction across group_elems ---
    stats_size = N * G
    n_tiles = max(1, (group_elems + BLOCK - 1) // BLOCK)  # number of partitions per (N,G)
    # partials shaped (stats_size, n_tiles)
    partial_sums = torch.empty((stats_size, n_tiles), device=x.device, dtype=x.dtype)
    partial_sumsqs = torch.empty((stats_size, n_tiles), device=x.device, dtype=x.dtype)

    grid = (stats_size * n_tiles,)
    _group_stats_partial_kernel[grid](
        x_contig, partial_sums.view(-1), partial_sumsqs.view(-1),
        N, C, S, G, group_elems, n_tiles,
        BLOCK=BLOCK
    )

    # finalize reduction: sum over tiles per (N,G)
    sums = partial_sums.sum(dim=1)
    sumsqs = partial_sumsqs.sum(dim=1)

    # compute mean and invstd on GPU (cheap: N*G elements)
    mean = sums / float(group_elems)
    var = sumsqs / float(group_elems) - mean * mean
    invstd = 1.0 / torch.sqrt(var + eps)

    # Prepare output
    out = torch.empty_like(x_contig)

    # --- Chunked normalization kernel: each program handles one TILE-sized chunk for a fixed (n,c) ---
    n_chunks = max(1, (S + BLOCK - 1) // BLOCK)
    grid2 = (N * C * n_chunks,)
    _groupnorm_affine_kernel_chunked[grid2](
        x_contig, out,
        mean, invstd,
        weight, bias,
        N, C, S, G, group_size, n_chunks,
        BLOCK=BLOCK
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized model: uses native ConvTranspose3d, then a Triton-fused ReLU+GroupNorm kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False, eps: float = 1e-5):
        super(ModelNew, self).__init__()
        # Keep the same conv transpose layer (leveraging highly optimized cuDNN/cuBLAS)
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, bias=bias)

        # Replace GroupNorm module with parameters used by the fused Triton kernel.
        # GroupNorm by default uses affine=True, so we register weight and bias parameters.
        self.groups = groups
        self.eps = eps
        self.out_channels = out_channels
        self.gn_weight = nn.Parameter(torch.ones(out_channels, dtype=torch.float32))
        self.gn_bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.float32))

    def forward(self, x):
        x = self.conv_transpose(x)
        # apply fused ReLU + GroupNorm (affine)
        out = _fused_relu_groupnorm(x, self.gn_weight, self.gn_bias, self.groups, eps=self.eps)
        return out