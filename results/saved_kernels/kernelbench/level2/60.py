import torch
import torch.nn as nn
import triton
import triton.language as tl

# Triton kernels to compute group moments (mean and variance) per (N, group)
@triton.jit
def _group_moments_kernel(
    x_ptr,               # pointer to input tensor (fp32)
    means_ptr,           # pointer to output means (fp32) length N * G
    vars_ptr,            # pointer to output variances (fp32) length N * G
    N, C, D, H, W, G,    # ints
    c_per_group,         # ints
    M,                   # total elements per group (c_per_group * D * H * W)
    BLOCK: tl.constexpr, # block size
):
    group_id = tl.program_id(0)  # each program handles one (n, group)
    total_groups = N * G
    if group_id >= total_groups:
        return

    n = group_id // G
    g = group_id % G

    # base linear offset for the group's first element in the contiguous layout
    # layout is N, C, D, H, W
    base = (n * C + g * c_per_group) * D * H * W

    offs = tl.arange(0, BLOCK)
    acc_sum = 0.0
    acc_sumsq = 0.0

    # loop over the group's M elements in chunks of BLOCK
    start = 0
    while start < M:
        idx = base + start + offs  # linear indices into x
        mask = idx < (base + M)
        vals = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        # compute Swish(vals) = vals * sigmoid(vals) so that moments are computed
        # on the activated values (matching model: Swish -> GroupNorm)
        sig = 1.0 / (1.0 + tl.exp(-vals))
        sw = vals * sig
        masked_sw = tl.where(mask, sw, 0.0)
        acc_sum += tl.sum(masked_sw)
        acc_sumsq += tl.sum(masked_sw * masked_sw)
        start += BLOCK

    # compute mean and variance
    M_f = M * 1.0
    mean = acc_sum / M_f
    var = acc_sumsq / M_f - mean * mean
    # store
    tl.store(means_ptr + group_id, mean)
    tl.store(vars_ptr + group_id, var)


# Triton kernel to apply: Swish -> GroupNorm (affine) -> HardSwish (fused)
@triton.jit
def _fused_gn_swish_hswish_kernel(
    x_ptr,       # input ptr (fp32)
    out_ptr,     # output ptr (fp32)
    means_ptr,   # per-(n,group) means (fp32)
    vars_ptr,    # per-(n,group) variances (fp32)
    weight_ptr,  # per-channel weight (fp32) length C
    bias_ptr,    # per-channel bias (fp32) length C
    N, C, D, H, W, G,
    c_per_group,
    M_total,     # total elements = N*C*D*H*W
    eps,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * BLOCK
    offs = start + tl.arange(0, BLOCK)
    mask = offs < M_total

    # load input values
    vals = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # compute indices: n, c
    DHW = D * H * W
    channel_block = C * DHW  # elements per sample across all channels
    n = offs // channel_block
    rem = offs - n * channel_block
    c = rem // DHW
    # group id per element: (n * G) + (c // c_per_group)
    grp = n * G + (c // c_per_group)

    # load mean and var per element's group
    mean = tl.load(means_ptr + grp, mask=mask, other=0.0)
    var = tl.load(vars_ptr + grp, mask=mask, other=0.0)

    # Swish: x * sigmoid(x) ; sigmoid = 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-vals))
    sw = vals * sig

    # invstd = 1 / sqrt(var + eps)
    invstd = 1.0 / tl.sqrt(var + eps)

    # normalize (using swish as input to groupnorm)
    normalized = (sw - mean) * invstd

    # load per-channel weight and bias
    w = tl.load(weight_ptr + c, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)

    affine = normalized * w + b

    # HardSwish: x * relu6(x + 3) / 6
    x_plus_3 = affine + 3.0
    relu6 = tl.minimum(tl.maximum(x_plus_3, 0.0), 6.0)
    hswish = affine * (relu6 * (1.0 / 6.0))

    # store output
    tl.store(out_ptr + offs, hswish, mask=mask)


class ModelNew(nn.Module):
    """
    Optimized Model:
      - Uses torch.nn.ConvTranspose3d for the transpose convolution (keeps PyTorch implementation)
      - Fuses Swish, GroupNorm (affine), and HardSwish into a single Triton kernel for speed.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super(ModelNew, self).__init__()
        # keep ConvTranspose3d in PyTorch
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        # Instead of using torch.nn.GroupNorm module, keep per-channel affine parameters and perform GN in Triton
        self.num_groups = groups
        self.eps = eps
        # affine parameters (initialized to the same defaults as GroupNorm: weight=1, bias=0)
        self.weight = nn.Parameter(torch.ones(out_channels, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.float32))

    def forward(self, x):
        # run conv transpose as-is
        x = self.conv_transpose(x)

        # ensure contiguous CUDA tensors
        if not x.is_cuda:
            # Triton kernels require CUDA tensors; move if needed
            x = x.cuda()
        x = x.contiguous()

        N, C, D, H, W = x.shape
        G = self.num_groups
        assert C % G == 0, "Number of channels must be divisible by number of groups"
        c_per_group = C // G
        M = c_per_group * D * H * W
        M_total = N * C * D * H * W

        # allocate temporary buffers for means and variances per (n, group)
        device = x.device
        dtype = x.dtype
        means = torch.empty((N * G,), device=device, dtype=dtype)
        vars_ = torch.empty((N * G,), device=device, dtype=dtype)

        # launch group moments kernel
        # blocks = number of (n, group)
        num_groups = N * G
        BLOCK_M = 1024  # block length for reductions (constexpr)
        grid = (num_groups,)
        # pass BLOCK as last positional argument
        _group_moments_kernel[grid](
            x,             # x_ptr
            means,         # means_ptr
            vars_,         # vars_ptr
            N, C, D, H, W, G,
            c_per_group,
            M,
            BLOCK_M
        )

        # allocate output
        out = torch.empty_like(x)

        # launch fused normalization + activations kernel
        total_elems = M_total
        BLOCK_E = 1024
        grid2 = ((total_elems + BLOCK_E - 1) // BLOCK_E,)
        _fused_gn_swish_hswish_kernel[grid2](
            x, out,
            means, vars_,
            self.weight, self.bias,
            N, C, D, H, W, G,
            c_per_group,
            total_elems,
            self.eps,
            BLOCK_E
        )

        return out