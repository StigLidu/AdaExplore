import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Autotune configurations tuned for A6000
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 512},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=3),
]


@triton.autotune(AUTOTUNE_CONFIGS, key=["B", "C", "S"])
@triton.jit
def _reduce_hardswish_kernel(
    x_ptr,           # pointer to flattened input tensor (B*C*S,)
    ch_sum_ptr,      # pointer to output sums per (B*C,)
    ch_sumsq_ptr,    # pointer to output sumsq per (B*C,)
    B,               # batch size
    C,               # channels
    S,               # spatial size (D*H*W)
    BLOCK: tl.constexpr,  # tile size
):
    # linear program id maps to (b * C + c)
    pid = tl.program_id(0)
    total = B * C
    if pid >= total:
        return

    # compute (b, c)
    b = pid // C
    c = pid - b * C  # pid % C

    base = pid * S  # flattened offset for (b, c, 0)

    acc = 0.0
    accsq = 0.0

    off = 0
    # loop over spatial dimension in tiles of size BLOCK
    while off < S:
        rng = tl.arange(0, BLOCK)
        offs = off + rng
        mask = offs < S

        addrs = base + offs  # absolute addresses into x_ptr
        vals = tl.load(x_ptr + addrs, mask=mask, other=0.0)

        # HardSwish: x * clamp(x + 3, 0, 6) / 6
        tmp = vals + 3.0
        tmp = tl.where(tmp < 0.0, 0.0, tmp)
        tmp = tl.where(tmp > 6.0, 6.0, tmp)
        hs = vals * (tmp * (1.0 / 6.0))

        acc += tl.sum(hs, axis=0)
        accsq += tl.sum(hs * hs, axis=0)

        off += BLOCK

    tl.store(ch_sum_ptr + pid, acc)
    tl.store(ch_sumsq_ptr + pid, accsq)


class ModelNew(nn.Module):
    """
    Optimized model:
      - Uses native PyTorch Conv3d (cuDNN) for the convolution.
      - Uses a Triton kernel that fuses HardSwish activation with reduction (sum and sumsq)
        across spatial dimensions for each (batch, channel), minimizing memory traffic.
      - Computes GroupNorm statistics from those reductions and returns the per-channel
        spatial mean after GroupNorm: output shape (B, C).
    """

    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super(ModelNew, self).__init__()
        # Keep convolution in PyTorch to leverage cuDNN performance
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)
        # Keep GroupNorm module for affine parameters and eps
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        # x: (B, C_in, D, H, W)
        # 1) convolution
        x = self.conv(x)  # (B, C, D, H, W)

        # Get shapes
        B, C, D, H, W = x.shape
        S = D * H * W

        # Ensure contiguous flattening for Triton kernel
        x_flat = x.contiguous().view(-1)

        # Allocate outputs for per-(B*C,) sums and sumsqs
        ch_sum = torch.empty(B * C, dtype=x.dtype, device=x.device)
        ch_sumsq = torch.empty(B * C, dtype=x.dtype, device=x.device)

        # Launch Triton kernel: one program per (b*c)
        grid = lambda meta: (B * C,)
        _reduce_hardswish_kernel[grid](x_flat, ch_sum, ch_sumsq, B, C, S)

        # reshape to (B, C)
        ch_sum = ch_sum.view(B, C)
        ch_sumsq = ch_sumsq.view(B, C)

        # GroupNorm parameters and shapes
        G = self.group_norm.num_groups
        eps = float(self.group_norm.eps)
        assert C % G == 0, "C must be divisible by num_groups"
        C_per_group = C // G
        N_per_group = C_per_group * S  # number of elements per group

        # Compute group sums and sumsq by reshaping and summing across channel-subgroup axis
        ch_sum_grouped = ch_sum.view(B, G, C_per_group)       # (B, G, C_per_group)
        ch_sumsq_grouped = ch_sumsq.view(B, G, C_per_group)   # (B, G, C_per_group)

        group_sum = ch_sum_grouped.sum(dim=2)       # (B, G)
        group_sumsq = ch_sumsq_grouped.sum(dim=2)   # (B, G)

        # Compute group mean and variance
        mean_g = group_sum / N_per_group            # (B, G)
        mean_sq_g = group_sumsq / N_per_group
        var_g = mean_sq_g - mean_g * mean_g        # (B, G)
        invstd_g = 1.0 / torch.sqrt(var_g + eps)   # (B, G)

        # Expand group stats back to per-channel (B, C)
        mean_g_exp = mean_g.repeat_interleave(C_per_group, dim=1)     # (B, C)
        invstd_g_exp = invstd_g.repeat_interleave(C_per_group, dim=1) # (B, C)

        # Per-channel spatial mean of the activated inputs
        ch_mean = ch_sum / S  # (B, C)

        # Affine params gamma (weight) and beta (bias) of GroupNorm
        weight = self.group_norm.weight
        bias = self.group_norm.bias
        if weight is None:
            weight = torch.ones(C, device=x.device, dtype=x.dtype)
        if bias is None:
            bias = torch.zeros(C, device=x.device, dtype=x.dtype)

        weight_b = weight.unsqueeze(0)  # (1, C)
        bias_b = bias.unsqueeze(0)      # (1, C)

        # Compute final per-channel spatial mean after GroupNorm:
        # mean_over_spatial( gamma * (hs - mean_group) * invstd_group + beta )
        # = gamma * invstd_group * (mean_hs - mean_group) + beta
        out = weight_b * invstd_g_exp * (ch_mean - mean_g_exp) + bias_b  # (B, C)

        return out


# === Test config (kept for compatibility) ===
batch_size = 1024
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 4

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]