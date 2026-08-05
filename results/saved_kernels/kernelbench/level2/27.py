import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Autotune configurations exploring BLOCK_S (spatial tile) and launch params.
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_S": 256},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_S": 256},  num_warps=8, num_stages=3),
    triton.Config({"BLOCK_S": 512},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_S": 512},  num_warps=8, num_stages=3),
    triton.Config({"BLOCK_S": 1024}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_S": 1024}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_S": 2048}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_S": 4096}, num_warps=8, num_stages=4),
]

@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['S', 'BLOCK_G'])
@triton.jit
def _gn_hs_pool_kernel(
    x_ptr,            # pointer to x flattened as (N, C, S) contiguous
    out_ptr,          # pointer to output (N, C) contiguous
    gamma_ptr,        # pointer to groupnorm weight (C,)
    beta_ptr,         # pointer to groupnorm bias (C,)
    N,                # number of samples
    C,                # number of channels
    S,                # spatial size = D*H*W
    eps,              # groupnorm eps (float)
    # constexpr block sizes
    BLOCK_G: tl.constexpr,  # channels per group (group_size)
    BLOCK_S: tl.constexpr,  # spatial block size for reduction (must be constexpr)
):
    # program ids
    n = tl.program_id(0)   # sample index
    g = tl.program_id(1)   # group index

    # channel indices for this group: c0 ... c0+BLOCK_G-1
    c0 = g * BLOCK_G
    cg = tl.arange(0, BLOCK_G)                        # (BLOCK_G,)
    c_idxs = c0 + cg                                  # (BLOCK_G,) channel indices

    # spatial arange (constexpr BLOCK_S)
    s_ar = tl.arange(0, BLOCK_S)                      # (BLOCK_S,)

    base_nc = n * C

    # per-channel accumulators across spatial dims
    sums = tl.zeros((BLOCK_G,), dtype=tl.float32)
    sumsq = tl.zeros((BLOCK_G,), dtype=tl.float32)

    s = 0
    # Unroll processing: try to handle up to 4 tiles per loop iteration when possible
    while s < S:
        # Tile 0
        s_idx = s + s_ar                               # (BLOCK_S,)
        mask_s = s_idx < S                             # (BLOCK_S,)

        c_idx_2d = c_idxs[:, None]                     # (BLOCK_G,1)
        s_idx_2d = s_idx[None, :]                      # (1,BLOCK_S)
        mask_c = c_idx_2d < C                          # (BLOCK_G,1)
        mask_2d = mask_c & (s_idx_2d < S)              # (BLOCK_G,BLOCK_S)

        offs = (base_nc + c_idx_2d) * S + s_idx_2d     # (BLOCK_G, BLOCK_S)
        vals = tl.load(x_ptr + offs, mask=mask_2d, other=0.0)  # (BLOCK_G, BLOCK_S)

        # HardSwish: x * relu6(x+3) / 6
        t = vals + 3.0
        t = tl.where(t > 0.0, t, 0.0)
        t = tl.where(t < 6.0, t, 6.0)
        y = vals * (t / 6.0)

        sums = sums + tl.sum(y, 1)
        sumsq = sumsq + tl.sum(y * y, 1)

        s += BLOCK_S

        # Tile 1
        if s < S:
            s_idx = s + s_ar
            s_idx_2d = s_idx[None, :]
            mask_2d = mask_c & (s_idx_2d < S)
            offs = (base_nc + c_idx_2d) * S + s_idx_2d
            vals = tl.load(x_ptr + offs, mask=mask_2d, other=0.0)

            t = vals + 3.0
            t = tl.where(t > 0.0, t, 0.0)
            t = tl.where(t < 6.0, t, 6.0)
            y = vals * (t / 6.0)

            sums = sums + tl.sum(y, 1)
            sumsq = sumsq + tl.sum(y * y, 1)

            s += BLOCK_S

        # Tile 2
        if s < S:
            s_idx = s + s_ar
            s_idx_2d = s_idx[None, :]
            mask_2d = mask_c & (s_idx_2d < S)
            offs = (base_nc + c_idx_2d) * S + s_idx_2d
            vals = tl.load(x_ptr + offs, mask=mask_2d, other=0.0)

            t = vals + 3.0
            t = tl.where(t > 0.0, t, 0.0)
            t = tl.where(t < 6.0, t, 6.0)
            y = vals * (t / 6.0)

            sums = sums + tl.sum(y, 1)
            sumsq = sumsq + tl.sum(y * y, 1)

            s += BLOCK_S

        # Tile 3
        if s < S:
            s_idx = s + s_ar
            s_idx_2d = s_idx[None, :]
            mask_2d = mask_c & (s_idx_2d < S)
            offs = (base_nc + c_idx_2d) * S + s_idx_2d
            vals = tl.load(x_ptr + offs, mask=mask_2d, other=0.0)

            t = vals + 3.0
            t = tl.where(t > 0.0, t, 0.0)
            t = tl.where(t < 6.0, t, 6.0)
            y = vals * (t / 6.0)

            sums = sums + tl.sum(y, 1)
            sumsq = sumsq + tl.sum(y * y, 1)

            s += BLOCK_S

    # compute group-level total sum and sumsq (scalars)
    total_sum = tl.sum(sums, 0)
    total_sumsq = tl.sum(sumsq, 0)

    K = BLOCK_G * S
    inv_K = 1.0 / (K * 1.0)
    mu = total_sum * inv_K
    var = total_sumsq * inv_K - mu * mu
    invstd = 1.0 / tl.sqrt(var + eps)

    # Load gamma and beta for this group's channels
    gammas = tl.load(gamma_ptr + c_idxs, mask=c_idxs < C, other=0.0)
    betas = tl.load(beta_ptr + c_idxs, mask=c_idxs < C, other=0.0)

    # compute per-channel mean across spatial positions
    per_channel_mean_spatial = sums / (S * 1.0)   # (BLOCK_G,)
    gain = gammas * invstd                        # (BLOCK_G,)
    res = (per_channel_mean_spatial - mu) * gain + betas

    out_offs = n * C + c_idxs
    tl.store(out_ptr + out_offs, res, mask=c_idxs < C)


def _fused_groupnorm_hs_mean(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, num_groups: int, eps: float):
    """
    Prepare inputs and launch the Triton kernel with autotuning.
    x: (N, C, D, H, W) float32 CUDA
    returns: (N, C) float32 CUDA
    """
    assert x.is_cuda and gamma.is_cuda and beta.is_cuda
    assert x.dtype == torch.float32 and gamma.dtype == torch.float32 and beta.dtype == torch.float32

    N, C, D, H, W = x.shape
    S = D * H * W
    group_size = C // num_groups
    # Flatten spatial dims: (N, C, S)
    x_flat = x.contiguous().view(N, C, S)
    out = torch.empty((N, C), device=x.device, dtype=x.dtype)

    # Heuristic to prefer larger BLOCK_S for big S but leave options to autotuner.
    # We pass BLOCK_G via autotune key as well; the autotuner will pick best BLOCK_S and launch params.
    grid = (N, num_groups)

    # Launch kernel (autotuned). BLOCK_G is constexpr and provided here.
    _gn_hs_pool_kernel[grid](
        x_flat,
        out,
        gamma.contiguous(),
        beta.contiguous(),
        N,
        C,
        S,
        float(eps),
        BLOCK_G=group_size,
        # BLOCK_S selected by autotuner configs (constexpr). No need to set here.
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized Model using Triton to fuse:
      - HardSwish activation
      - GroupNorm (per-group over channels and spatial elements)
      - Mean pooling across spatial dimensions (D,H,W) -> (B, C)

    Implementation notes:
      - We keep PyTorch's Conv3d (cuDNN) for convolution performance.
      - The post-convolution pipeline (hardswish + groupnorm + mean) is implemented
        in a highly-tuned Triton kernel with autotuning over spatial tile sizes and launch params.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)
        # Keep GroupNorm module so we can reuse its parameters and eps
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        # conv -> fused (hardswish + groupnorm + mean over spatial dims)
        x = self.conv(x)  # (N, C, D, H, W)
        # Ensure gamma and beta exist (GroupNorm default affine=True)
        if self.group_norm.weight is None:
            gamma = torch.ones(self.group_norm.num_channels, device=x.device, dtype=x.dtype)
        else:
            gamma = self.group_norm.weight
        if self.group_norm.bias is None:
            beta = torch.zeros(self.group_norm.num_channels, device=x.device, dtype=x.dtype)
        else:
            beta = self.group_norm.bias

        return _fused_groupnorm_hs_mean(x, gamma, beta, self.group_norm.num_groups, float(self.group_norm.eps))