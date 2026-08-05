import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Reduction kernel: compute partial sums and sum-of-squares of GELU(x) per (N,G) group and atomically accumulate.
@triton.jit
def _group_sum_squares_kernel(
    x_ptr,           # *ptr to input tensor (N, C, H, W)
    sum_ptr,         # *ptr to accumulated sum (N*G,)
    sumsq_ptr,       # *ptr to accumulated sumsq (N*G,)
    N, C, H, W, G, C_per_group, L,  # ints
    BLOCK: tl.constexpr,
):
    n = tl.program_id(0)
    g = tl.program_id(1)
    block_idx = tl.program_id(2)

    block_start = block_idx * BLOCK
    offs = block_start + tl.arange(0, BLOCK)
    mask = offs < L

    spatial_per_channel = H * W
    channel_idx = offs // spatial_per_channel
    spatial_idx = offs - channel_idx * spatial_per_channel

    c = g * C_per_group + channel_idx  # shape [BLOCK]

    base_nc = n * C
    addr = (base_nc + c) * spatial_per_channel + spatial_idx

    # Load x values (masked)
    x_vals = tl.load(x_ptr + addr, mask=mask, other=0.0)

    # Compute GELU in-kernel using erf: gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865476
    gelu_vals = 0.5 * x_vals * (1.0 + tl.erf(x_vals * inv_sqrt2))

    # partial sums over the block (masked positions already zeroed)
    partial_sum = tl.sum(gelu_vals)
    partial_sumsq = tl.sum(gelu_vals * gelu_vals)

    # atomic accumulate into global accumulators for (n,g)
    idx = n * G + g
    tl.atomic_add(sum_ptr + idx, partial_sum)
    tl.atomic_add(sumsq_ptr + idx, partial_sumsq)


# Finalize kernel: compute mean and rstd from accumulated sum and sumsq
@triton.jit
def _group_finalize_kernel(
    sum_ptr,         # *ptr to accumulated sum (N*G,)
    sumsq_ptr,       # *ptr to accumulated sumsq (N*G,)
    mean_ptr,        # *ptr to output mean (N*G,)
    rstd_ptr,        # *ptr to output rstd (N*G,)
    N, G, L, eps,    # ints/floats
):
    n = tl.program_id(0)
    g = tl.program_id(1)
    idx = n * G + g

    s = tl.load(sum_ptr + idx)
    ss = tl.load(sumsq_ptr + idx)

    mean = s / L
    var = ss / L - mean * mean
    # numerical stability: var might be slightly negative due to fp accumulation, clamp to >= 0
    var = tl.where(var > 0.0, var, 0.0)
    rstd = 1.0 / tl.sqrt(var + eps)

    tl.store(mean_ptr + idx, mean)
    tl.store(rstd_ptr + idx, rstd)


# Apply kernel: load x, compute GELU, normalize using mean/rstd, apply affine (gamma/beta)
@triton.jit
def _groupnorm_affine_kernel(
    x_ptr,           # *ptr to input tensor (N, C, H, W)
    out_ptr,         # *ptr to output tensor (N, C, H, W)
    gamma_ptr,       # *ptr to weight (C,)
    beta_ptr,        # *ptr to bias (C,)
    mean_ptr,        # *ptr to mean (N * G,)
    rstd_ptr,        # *ptr to rstd (N * G,)
    N, C, H, W, G, C_per_group, L,  # ints
    BLOCK: tl.constexpr,
):
    n = tl.program_id(0)
    g = tl.program_id(1)
    block_idx = tl.program_id(2)

    block_start = block_idx * BLOCK
    offs = block_start + tl.arange(0, BLOCK)
    mask = offs < L

    spatial_per_channel = H * W
    channel_idx = offs // spatial_per_channel
    spatial_idx = offs - channel_idx * spatial_per_channel

    c = g * C_per_group + channel_idx  # shape [BLOCK]

    base_nc = n * C
    addr = (base_nc + c) * spatial_per_channel + spatial_idx

    x_vals = tl.load(x_ptr + addr, mask=mask, other=0.0)

    # compute GELU in-kernel (fused)
    inv_sqrt2 = 0.7071067811865476
    gelu_vals = 0.5 * x_vals * (1.0 + tl.erf(x_vals * inv_sqrt2))

    gamma = tl.load(gamma_ptr + c, mask=mask, other=1.0)
    beta = tl.load(beta_ptr + c, mask=mask, other=0.0)

    mean_idx = n * G + g
    mean_val = tl.load(mean_ptr + mean_idx)
    rstd_val = tl.load(rstd_ptr + mean_idx)

    out = (gelu_vals - mean_val) * rstd_val * gamma + beta

    tl.store(out_ptr + addr, out, mask=mask)


def triton_groupnorm_affine(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, num_groups: int):
    """
    New two-stage Triton-backed groupnorm that:
      - computes GELU(x) reductions (sum and sumsq) per (N,G) on-device,
      - finalizes mean and rstd on-device,
      - applies normalization+affine with GELU fused into the apply kernel.

    Inputs:
      x: (N,C,H,W) CUDA float32
      weight: (C,)
      bias: (C,)
      num_groups: G
    Returns:
      out: normalized tensor (N,C,H,W)
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "All tensors must be on CUDA."
    assert x.dtype == torch.float32, "Only fp32 supported in this kernel."

    N, C, H, W = x.shape
    G = num_groups
    assert C % G == 0, "num_groups must divide num_channels"
    C_per_group = C // G
    L = C_per_group * H * W

    x = x.contiguous()
    out = torch.empty_like(x)
    weight = weight.contiguous()
    bias = bias.contiguous()

    # allocate accumulators for sum and sumsq (N*G)
    device = x.device
    sum_buf = torch.zeros((N * G,), dtype=torch.float32, device=device)
    sumsq_buf = torch.zeros((N * G,), dtype=torch.float32, device=device)

    # reduction kernel params
    BLOCK_REDUCE = 512  # constexpr tuneable
    num_blocks = (L + BLOCK_REDUCE - 1) // BLOCK_REDUCE
    grid_reduce = (N, G, num_blocks)

    # launch reduction kernel to accumulate GELU sums and sumsq
    _group_sum_squares_kernel[grid_reduce](
        x, sum_buf, sumsq_buf,
        N, C, H, W, G, C_per_group, L, BLOCK_REDUCE,
    )

    # finalize kernel: compute mean and rstd per (N,G)
    mean_buf = torch.empty((N * G,), dtype=torch.float32, device=device)
    rstd_buf = torch.empty((N * G,), dtype=torch.float32, device=device)
    eps = float(1e-5)  # default small eps; will override with module eps when used below

    grid_finalize = (N, G)
    _group_finalize_kernel[grid_finalize](
        sum_buf, sumsq_buf, mean_buf, rstd_buf,
        N, G, L, eps,
    )

    return out, mean_buf, rstd_buf  # returns intermediate buffers; higher-level caller will call apply kernel


class ModelNew(nn.Module):
    """
    Optimized model that uses PyTorch ConvTranspose2d and Triton-backed
    fused GroupNorm affine application. GELU and reductions are performed
    on-device to reduce memory traffic.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super(ModelNew, self).__init__()
        # Keep ConvTranspose2d for correctness and leveraging cuDNN
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        # Keep a GroupNorm module to hold learnable parameters (weight, bias) and eps
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        # Keep groups param for compatibility (not used directly here, but preserved)
        self.groups = groups
        self.num_groups = num_groups

    def forward(self, x):
        # conv transpose (use PyTorch implementation)
        x = self.conv_transpose(x)

        # We'll perform GELU and reductions/apply in Triton (fused).
        N, C, H, W = x.shape
        G = self.num_groups
        assert C % G == 0, "num_groups must divide num_channels"
        C_per_group = C // G
        L = C_per_group * H * W

        # run reduction + finalize to compute mean and rstd on-device (also fused GELU)
        x = x.contiguous()
        out = torch.empty_like(x)
        weight = self.group_norm.weight.contiguous()
        bias = self.group_norm.bias.contiguous()

        device = x.device

        # Allocate accumulators for reduction and buffers for final stats
        sum_buf = torch.zeros((N * G,), dtype=torch.float32, device=device)
        sumsq_buf = torch.zeros((N * G,), dtype=torch.float32, device=device)

        # Reduction kernel params
        BLOCK_REDUCE = 512
        num_blocks = (L + BLOCK_REDUCE - 1) // BLOCK_REDUCE
        grid_reduce = (N, G, num_blocks)

        # Launch reduction kernel
        _group_sum_squares_kernel[grid_reduce](
            x, sum_buf, sumsq_buf,
            N, C, H, W, G, C_per_group, L, BLOCK_REDUCE,
        )

        # Finalize mean and rstd
        mean_buf = torch.empty((N * G,), dtype=torch.float32, device=device)
        rstd_buf = torch.empty((N * G,), dtype=torch.float32, device=device)
        eps = float(self.group_norm.eps)
        grid_finalize = (N, G)
        _group_finalize_kernel[grid_finalize](
            sum_buf, sumsq_buf, mean_buf, rstd_buf,
            N, G, L, eps,
        )

        # Apply kernel params
        BLOCK_APPLY = 512
        num_blocks_apply = (L + BLOCK_APPLY - 1) // BLOCK_APPLY
        grid_apply = (N, G, num_blocks_apply)

        # Launch apply kernel (GELU fused inside)
        _groupnorm_affine_kernel[grid_apply](
            x, out, weight, bias, mean_buf, rstd_buf,
            N, C, H, W, G, C_per_group, L, BLOCK_APPLY,
        )
        return out


# Helper functions for the environment that generate inputs (kept for compatibility)
batch_size   = 128
in_channels  = 64
out_channels = 64
height = width = 256
kernel_size  = 3
stride       = 1
groups = 8
num_groups = 8

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, groups, num_groups]