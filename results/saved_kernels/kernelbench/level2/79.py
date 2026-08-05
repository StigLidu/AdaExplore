import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Autotune configurations chosen for NVIDIA A6000 (Ampere).
MEAN_INVSTD_AUTOTUNE = [
    triton.Config({"BLOCK_S": 256},  num_warps=4,  num_stages=2),
    triton.Config({"BLOCK_S": 512},  num_warps=4,  num_stages=2),
    triton.Config({"BLOCK_S": 1024}, num_warps=8,  num_stages=3),
    triton.Config({"BLOCK_S": 2048}, num_warps=8,  num_stages=3),
    triton.Config({"BLOCK_S": 4096}, num_warps=16, num_stages=3),
]

MAX_CHANNEL_AUTOTUNE = [
    # Add a small BLOCK_C constexpr so channel-blocking can be tuned (C is small; set 16 for this model).
    triton.Config({"BLOCK_S": 128,  "BLOCK_C": 16},  num_warps=2,  num_stages=2),
    triton.Config({"BLOCK_S": 256,  "BLOCK_C": 16},  num_warps=4,  num_stages=2),
    triton.Config({"BLOCK_S": 512,  "BLOCK_C": 16},  num_warps=8,  num_stages=3),
    triton.Config({"BLOCK_S": 1024, "BLOCK_C": 16},  num_warps=8,  num_stages=3),
    triton.Config({"BLOCK_S": 2048, "BLOCK_C": 16},  num_warps=16, num_stages=3),
]


@triton.autotune(MEAN_INVSTD_AUTOTUNE, key=["N", "C", "S"])
@triton.jit
def _compute_mean_invstd_kernel(
    x_ptr,        # pointer to input flattened (N*C*S)
    mean_ptr,     # pointer to output mean flattened (N*C,)
    invstd_ptr,   # pointer to output invstd flattened (N*C,)
    N, C, S,
    eps: tl.constexpr,
    BLOCK_S: tl.constexpr
):
    # program indices over (n, c)
    n = tl.program_id(0)
    c = tl.program_id(1)
    row = n * C + c
    row_start = row * S

    offs = tl.arange(0, BLOCK_S)

    # vector accumulators in registers
    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)
    accsq = tl.zeros((BLOCK_S,), dtype=tl.float32)

    # iterate over spatial dimension in blocks
    for s_start in range(0, S, BLOCK_S):
        idx = row_start + s_start + offs
        mask = (s_start + offs) < S
        x = tl.load(x_ptr + idx, mask=mask, other=0.0)
        acc += x
        accsq += x * x

    # horizontal reduce
    sum_val = tl.sum(acc)
    sumsq = tl.sum(accsq)

    mean = sum_val / S
    var = sumsq / S - mean * mean
    invstd = 1.0 / tl.sqrt(var + eps)

    tl.store(mean_ptr + row, mean)
    tl.store(invstd_ptr + row, invstd)


@triton.autotune(MAX_CHANNEL_AUTOTUNE, key=["N", "C", "S"])
@triton.jit
def _max_over_channels_kernel(
    x_ptr,       # pointer to input flattened (N*C*S)
    mean_ptr,    # pointer to mean flattened (N*C,)
    invstd_ptr,  # pointer to invstd flattened (N*C,)
    mult_ptr,    # pointer to multiplier (C,)
    out_ptr,     # pointer to output flattened (N*S,)
    N, C, S,
    clamp_min: tl.constexpr,
    clamp_max: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_C: tl.constexpr
):
    n = tl.program_id(0)
    s_block = tl.program_id(1)
    s_start = s_block * BLOCK_S

    offs = tl.arange(0, BLOCK_S)
    s_idx = s_start + offs
    mask = s_idx < S

    # initialize max with very small values
    neg_inf = -3.4e38
    max_val = tl.full((BLOCK_S,), neg_inf, dtype=tl.float32)

    # Precompute a row base for this (n,*) program to reduce repeated multiplications in the inner loop
    row_base = n * C

    # iterate over channels (C small)
    # We move the per-channel multiplication into the normalization factor:
    # val = clamp((x - mean) * (invstd * mult), bounds) which is equivalent to clamp(normed, a, b) * mult
    for c in range(0, C):
        row = row_base + c
        mean = tl.load(mean_ptr + row)
        invstd = tl.load(invstd_ptr + row)
        mult = tl.load(mult_ptr + c)

        # precompute combined scalar and per-channel clamping bounds
        invstd_mul = invstd * mult
        b1 = clamp_min * mult
        b2 = clamp_max * mult
        lo = tl.minimum(b1, b2)
        hi = tl.maximum(b1, b2)

        idx = row * S + s_idx
        x = tl.load(x_ptr + idx, mask=mask, other=0.0)

        # compute scaled normalized value in one multiply, then clamp
        val = (x - mean) * invstd_mul
        val = tl.maximum(val, lo)
        val = tl.minimum(val, hi)

        # update running max across channels
        max_val = tl.where(val > max_val, val, max_val)

    out_idx = n * S + s_start + offs
    tl.store(out_ptr + out_idx, max_val, mask=mask)


def triton_compute_mean_invstd(x: torch.Tensor, eps: float = 1e-5):
    """
    x: tensor shape (N, C, S) viewable (contiguous in memory with S as fastest dim).
    Returns mean and invstd tensors of shape (N*C,) on device.
    """
    assert x.is_cuda, "Input must be on CUDA."
    assert x.dtype == torch.float32, "fp32 required"

    N, C, S = x.shape
    x_flat = x.reshape(-1)

    mean = torch.empty((N * C,), dtype=torch.float32, device=x.device)
    invstd = torch.empty((N * C,), dtype=torch.float32, device=x.device)

    grid = (N, C)
    _compute_mean_invstd_kernel[grid](x_flat, mean, invstd, N, C, S, float(eps))
    return mean, invstd


def triton_compute_channel_max(x: torch.Tensor, mean: torch.Tensor, invstd: torch.Tensor, multiplier: torch.Tensor, clamp_min: float, clamp_max: float):
    """
    For each (n, s) compute max over channels after normalization, clamp, and final multiplication.
    This implementation multiplies the invstd by the per-channel multiplier first, then clamps the scaled value.
    That reduces elementwise multiplies and allows precomputing channel-specific clamp bounds.
    x: (N, C, S) viewable. mean/invstd: (N*C,), multiplier: (C,)
    Returns out tensor of shape (N, S).
    """
    assert x.is_cuda and mean.is_cuda and invstd.is_cuda and multiplier.is_cuda, "All tensors must be on CUDA."
    assert x.dtype == torch.float32

    N, C, S = x.shape
    x_flat = x.reshape(-1)

    out = torch.empty((N * S,), dtype=torch.float32, device=x.device)
    mult_flat = multiplier.view(-1).contiguous()

    # lambda grid to allow autotuner selection of BLOCK_S
    grid = lambda meta: (N, (S + meta["BLOCK_S"] - 1) // meta["BLOCK_S"])

    _max_over_channels_kernel[grid](x_flat, mean, invstd, mult_flat, out, N, C, S, float(clamp_min), float(clamp_max))
    return out.view(N, S)


class ModelNew(nn.Module):
    """
    Optimized Model using Triton kernels.

    Optimizations implemented:
      - Fuse the first elementwise multiply (x * multiplier) into conv weights to avoid a large post-conv elementwise kernel.
      - Compute InstanceNorm statistics (mean & invstd per (n,c)) using an efficient Triton reduction kernel.
      - Compute clamp, second multiplication, and channel-wise max in a second Triton kernel,
        with the per-channel multiplier folded into the normalization factor to reduce per-element multiplies.
      - Autotuned BLOCK_S and warp configurations for the A6000.
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        # multiplier used twice in original model
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        # match typical InstanceNorm eps
        self.eps = 1e-5

    def forward(self, x):
        # x: (N, in_channels, D, H, W)
        assert x.is_cuda, "Input must be on CUDA."

        # Fuse first multiplication into conv weights to avoid an elementwise multiply over a large tensor.
        C = self.conv.out_channels
        mult_w = self.multiplier.view(C, 1, 1, 1, 1)
        x = F.conv3d(x, self.conv.weight * mult_w, bias=self.conv.bias,
                     stride=self.conv.stride, padding=self.conv.padding,
                     dilation=self.conv.dilation, groups=self.conv.groups)

        # shapes and flatten spatial dims
        N, C, D, H, W = x.shape
        S = D * H * W
        x_nc_s = x.view(N, C, S)

        # Kernel A: compute per-(n,c) mean & invstd
        mean, invstd = triton_compute_mean_invstd(x_nc_s, eps=self.eps)

        # Kernel B: compute per-(n,s) max over channels after normalization, clamp, and second multiply
        out_ns = triton_compute_channel_max(x_nc_s, mean, invstd, self.multiplier.view(C), self.clamp_min, self.clamp_max)

        # reshape back to (N, D, H, W)
        out = out_ns.view(N, D, H, W)
        return out