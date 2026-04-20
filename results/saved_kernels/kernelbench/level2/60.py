import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configs for our Triton kernels
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 256},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 512},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2),
]

@triton.autotune(
    configs=AUTOTUNE_CONFIGS,
    key=['N', 'C', 'S', 'channels_per_group']
)
@triton.jit
def _compute_group_mean_invstd_kernel(
    x_ptr,            # pointer to flattened input tensor (N*C*S,)
    mean_ptr,         # pointer to output mean (N*groups,)
    invstd_ptr,       # pointer to output invstd (N*groups,)
    N, C, S, groups, channels_per_group, eps,
    BLOCK_SIZE: tl.constexpr
):
    """
    Each program handles one (n, g) pair (one group in one batch element)
    and reduces over channels_in_group * S elements computing mean and invstd
    of the Swish-activated values.
    """
    pid = tl.program_id(0)  # 0 .. N*groups-1
    n = pid // groups
    g = pid % groups

    # total number of elements to reduce for this (n,g)
    total = channels_per_group * S

    # base offset in the flattened tensor: ((n * C) + c_start) * S
    c_start = g * channels_per_group
    base = (n * C + c_start) * S

    offs = tl.arange(0, BLOCK_SIZE)
    acc = 0.0
    acc2 = 0.0

    # Loop over the chunk of data for this group
    start = 0
    while start < total:
        idx = base + start + offs
        mask = (start + offs) < total
        vals = tl.load(x_ptr + idx, mask=mask, other=0.0)

        # Swish activation: x * sigmoid(x)
        sig = 1.0 / (1.0 + tl.exp(-vals))
        vs = vals * sig

        # sum and sum of squares across the block lanes
        acc = acc + tl.sum(vs, axis=0)
        acc2 = acc2 + tl.sum(vs * vs, axis=0)

        start += BLOCK_SIZE

    elems = total
    mean = acc / elems
    var = acc2 / elems - mean * mean
    invstd = 1.0 / tl.sqrt(var + eps)

    out_idx = n * groups + g
    # store scalar mean and invstd
    tl.store(mean_ptr + out_idx, mean)
    tl.store(invstd_ptr + out_idx, invstd)


@triton.autotune(
    configs=AUTOTUNE_CONFIGS,
    key=['N', 'C', 'S', 'channels_per_group']
)
@triton.jit
def _apply_groupnorm_and_activations_kernel(
    x_ptr, out_ptr,          # flattened input and output pointers (N*C*S,)
    mean_ptr, invstd_ptr,    # (N*groups,)
    weight_ptr, bias_ptr,    # (C,), per-channel affine parameters
    N, C, S, groups, channels_per_group,
    BLOCK_SIZE: tl.constexpr
):
    """
    Apply group normalization (using precomputed mean & invstd) followed by
    affine (weight, bias) and then HardSwish activation. Swish has already
    been used to compute the group statistics; we apply Swish again to compute
    the normalized values correctly per-element.
    """
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    n_elems = N * C * S
    mask = offs < n_elems

    vals = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # compute n, c, s from flattened index
    # n = offs // (C*S)
    # rem = offs - n*(C*S)
    # c = rem // S
    # group g = c // channels_per_group
    denom = C * S
    n_idx = offs // denom
    rem = offs - n_idx * denom
    c_idx = rem // S
    g_idx = c_idx // channels_per_group
    group_index = n_idx * groups + g_idx

    # load mean and invstd for each lane
    mean_g = tl.load(mean_ptr + group_index, mask=mask, other=0.0)
    invstd_g = tl.load(invstd_ptr + group_index, mask=mask, other=0.0)

    # recompute Swish for elementwise value (same as used in reduction)
    sig = 1.0 / (1.0 + tl.exp(-vals))
    vs = vals * sig

    # Normalize
    norm = (vs - mean_g) * invstd_g

    # load per-channel affine params
    w = tl.load(weight_ptr + c_idx, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

    out = norm * w + b

    # HardSwish: x * clamp(x + 3, 0, 6) / 6
    t = out + 3.0
    # clamp using nested where
    t = tl.where(t <= 0.0, 0.0, t)
    t = tl.where(t >= 6.0, 6.0, t)
    out = out * (t / 6.0)

    tl.store(out_ptr + offs, out, mask=mask)


def triton_groupnorm_swish_hardswish(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, groups: int, eps: float):
    """
    Wrapper that runs the Triton kernels to:
      1) compute group means and invstd on Swish-activated conv outputs
      2) apply group normalization, affine transform, and HardSwish elementwise
    """
    assert x.is_cuda, "Input must be on CUDA"
    x = x.contiguous()
    N, C, D, H, W = x.shape
    S = D * H * W
    channels_per_group = C // groups

    x_flat = x.view(-1)

    # allocate outputs for mean and invstd
    mean = torch.empty((N * groups,), device=x.device, dtype=x.dtype)
    invstd = torch.empty((N * groups,), device=x.device, dtype=x.dtype)

    # Launch reduction kernel to compute mean and invstd per (n, g)
    nprog = N * groups
    grid_reduce = lambda meta: (nprog,)
    _compute_group_mean_invstd_kernel[grid_reduce](x_flat, mean, invstd, N, C, S, groups, channels_per_group, eps)

    # allocate output tensor
    out = torch.empty_like(x_flat)

    # Launch elementwise kernel to apply normalization, affine, and HardSwish
    n_elements = x_flat.numel()
    grid_apply = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _apply_groupnorm_and_activations_kernel[grid_apply](
        x_flat, out, mean, invstd, weight.contiguous(), bias.contiguous(),
        N, C, S, groups, channels_per_group
    )

    return out.view(N, C, D, H, W)


class ModelNew(nn.Module):
    """
    Optimized model:
      - keeps PyTorch ConvTranspose3d for correctness and complexity
      - replaces Swish + GroupNorm + HardSwish with a fused Triton implementation
        that computes per-group statistics on Swish-activated values and then
        applies GroupNorm + affine + HardSwish in a single pass.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        # emulate GroupNorm parameters: weight (gamma) and bias (beta)
        self.gn_weight = nn.Parameter(torch.ones(out_channels, dtype=torch.float32))
        self.gn_bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.float32))
        self.groups = groups
        self.eps = eps

    def forward(self, x):
        # conv transpose (kept as PyTorch op)
        x = self.conv_transpose(x)
        # fused Triton kernel: Swish -> GroupNorm (with affine) -> HardSwish
        x = triton_groupnorm_swish_hardswish(x, self.gn_weight, self.gn_bias, self.groups, self.eps)
        return x