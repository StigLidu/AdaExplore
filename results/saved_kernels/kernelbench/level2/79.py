import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configurations for the reduction (stats) kernel
_INST_STATS_CONFIGS = [
    triton.Config({"BLOCK": 256}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK": 512}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=3),
]

# Autotune configurations for the apply kernel (normalization + clamp + post-mul + channel max)
_INST_APPLY_CONFIGS = [
    triton.Config({"BLOCK": 64},  num_warps=2, num_stages=2),
    triton.Config({"BLOCK": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 256}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK": 512}, num_warps=8, num_stages=3),
]


@triton.autotune(
    configs=_INST_STATS_CONFIGS,
    key=["N", "C", "S"],
)
@triton.jit
def _instancenorm_stats_kernel(
    input_ptr,        # pointer to input tensor (N*C*S flattened)
    mult_ptr,         # pointer to multiplier per channel (C,)
    mean_ptr,         # pointer to output mean (N*C,)
    invstd_ptr,       # pointer to output invstd (N*C,)
    N, C, S, eps,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)   # one program per (n, c)
    n = pid // C
    c = pid % C
    # start index for this (n,c) channel in flattened input
    channel_start = (n * C + c) * S

    offs = tl.arange(0, BLOCK)
    # accumulators initialized to 0.0
    acc_sum = tl.sum(tl.zeros((BLOCK,), dtype=tl.float32), 0)
    acc_sumsq = tl.sum(tl.zeros((BLOCK,), dtype=tl.float32), 0)

    mult_val = tl.load(mult_ptr + c)  # scalar multiplier for this channel

    s_ptr = 0
    while s_ptr < S:
        off = s_ptr + offs
        mask = off < S
        ptrs = input_ptr + channel_start + off
        vals = tl.load(ptrs, mask=mask, other=0.0)
        # pre-normalization multiplication
        vals = vals * mult_val
        acc_sum = acc_sum + tl.sum(vals, 0)
        acc_sumsq = acc_sumsq + tl.sum(vals * vals, 0)
        s_ptr += BLOCK

    mean = acc_sum / S
    var = acc_sumsq / S - mean * mean
    invstd = 1.0 / tl.sqrt(var + eps)

    out_idx = n * C + c
    tl.store(mean_ptr + out_idx, mean)
    tl.store(invstd_ptr + out_idx, invstd)


@triton.autotune(
    configs=_INST_APPLY_CONFIGS,
    key=["N", "C", "S", "clamp_min", "clamp_max"],
)
@triton.jit
def _instancenorm_apply_kernel(
    input_ptr,        # pointer to input tensor (N*C*S flattened)
    mult_ptr,         # pointer to multiplier per channel (C,)
    mean_ptr,         # pointer to mean (N*C,)
    invstd_ptr,       # pointer to invstd (N*C,)
    out_ptr,          # pointer to output tensor (N*S) flattened
    N, C, S, clamp_min, clamp_max,
    BLOCK: tl.constexpr,
):
    n = tl.program_id(0)
    sb = tl.program_id(1)  # spatial block index
    offs = sb * BLOCK + tl.arange(0, BLOCK)
    mask = offs < S

    # initialize accumulator to very small number for max reduction
    acc = tl.full((BLOCK,), -1e9, dtype=tl.float32)

    # iterate over channels - small C (e.g., 16) makes this efficient
    for c in range(C):
        channel_start = (n * C + c) * S
        ptrs = input_ptr + channel_start + offs
        vals = tl.load(ptrs, mask=mask, other=0.0)

        mult_val = tl.load(mult_ptr + c)            # scalar
        vals = vals * mult_val                      # first multiplication (pre-norm)

        idx = n * C + c
        mean = tl.load(mean_ptr + idx)              # scalar
        invstd = tl.load(invstd_ptr + idx)          # scalar

        normalized = (vals - mean) * invstd

        # clamp
        clamped = tl.maximum(normalized, clamp_min)
        clamped = tl.minimum(clamped, clamp_max)

        # second multiplication (post-clamp)
        out_vals = clamped * mult_val

        # update max accumulator
        acc = tl.maximum(acc, out_vals)

    out_ptrs = out_ptr + n * S + offs
    tl.store(out_ptrs, acc, mask=mask)


def _compute_instance_stats(x: torch.Tensor, multiplier: torch.Tensor, eps: float = 1e-5):
    """
    Computes mean and invstd per (N, C) for x * multiplier using Triton kernel.
    x: (N, C, D, H, W) contiguous tensor
    multiplier: (C,) or (C,1,1,1) tensor
    Returns mean (N, C) and invstd (N, C)
    """
    assert x.is_cuda and multiplier.is_cuda
    N, C, D, H, W = x.shape
    S = D * H * W

    x_flat = x.view(-1)
    mult_flat = multiplier.view(-1).contiguous()

    mean = torch.empty((N, C), dtype=torch.float32, device=x.device)
    invstd = torch.empty((N, C), dtype=torch.float32, device=x.device)

    # grid: one program per (N*C)
    grid = lambda meta: ((N * C + meta["BLOCK"] - 1) // meta["BLOCK"],)

    _instancenorm_stats_kernel[grid](
        x_flat,
        mult_flat,
        mean,
        invstd,
        N, C, S, float(eps)
    )
    return mean, invstd


def _apply_instance_norm_and_reduce(x: torch.Tensor, multiplier: torch.Tensor, mean: torch.Tensor, invstd: torch.Tensor, clamp_min: float, clamp_max: float):
    """
    Applies instance normalization using precomputed mean/invstd, then clamps, multiplies by multiplier, and reduces max over channels.
    Returns tensor of shape (N, D, H, W)
    """
    assert x.is_cuda and multiplier.is_cuda and mean.is_cuda and invstd.is_cuda
    N, C, D, H, W = x.shape
    S = D * H * W

    x_flat = x.view(-1)
    mult_flat = multiplier.view(-1).contiguous()

    out = torch.empty((N, S), dtype=torch.float32, device=x.device)

    # grid: N x n_blocks over spatial dimension
    grid = lambda meta: (N, (S + meta["BLOCK"] - 1) // meta["BLOCK"])

    _instancenorm_apply_kernel[grid](
        x_flat,
        mult_flat,
        mean,
        invstd,
        out,
        N, C, S, float(clamp_min), float(clamp_max)
    )

    out = out.view(N, D, H, W)
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the original Model using Triton kernels to perform:
      - instance normalization statistics (mean & invstd) of (conv_output * multiplier)
      - normalization, clamp, second multiplication, and channel-wise max reduction
    The 3D convolution is kept as PyTorch Conv3d to leverage cuDNN.
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        # store multiplier as a parameter with given shape (e.g., (C,1,1,1))
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.eps = 1e-5

    def forward(self, x):
        # x: (N, in_channels, D, H, W)
        x = self.conv(x)

        # Ensure contiguous and fp32
        x = x.contiguous().to(dtype=torch.float32)
        # multiplier to shape (C,)
        multiplier = self.multiplier.view(self.multiplier.shape[0]).contiguous().to(dtype=torch.float32)

        # Compute per-(N,C) mean and invstd for (x * multiplier)
        mean, invstd = _compute_instance_stats(x, multiplier, eps=self.eps)

        # Apply normalization, clamp, second multiply, and reduce max over channels
        out = _apply_instance_norm_and_reduce(x, multiplier, mean, invstd, self.clamp_min, self.clamp_max)

        return out