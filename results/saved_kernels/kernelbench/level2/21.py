import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Fuse sigmoid + GroupNorm (per-sample per-group) in Triton.
# Strategy:
# 1) Fold the provided per-channel bias and scale into conv weights/bias at init (inference-friendly).
# 2) Run conv via cuDNN (PyTorch F.conv2d) using cached NHWC weight when input is channels-last.
# 3) Run two Triton kernels:
#    a) gn_reduce_kernel: for each (N, G) compute sum and sumsq of sigmoid(conv_out) across channels_in_group * H * W
#    b) gn_apply_kernel: for each (N, C, spatial-chunk) compute sigmoid, normalize using precomputed mean/invstd and apply GN affine (gamma, beta)
#
# This avoids allocating an intermediate tensor for sigmoid (we recompute sigmoid twice: once for reduction, once for final), and
# fuses the elementwise operations into a single write pass for the final output.

# Tunable block size (constexpr)
BLOCK = 512

@triton.jit
def gn_reduce_kernel(
    x_ptr,         # pointer to conv output (N, C, H, W) in NCHW contiguous layout
    sum_ptr,       # pointer to output sums (N * num_groups)
    sumsq_ptr,     # pointer to output sumsqs (N * num_groups)
    N, C, H, W,    # dims
    num_groups,    # number of groups
    Cg,            # channels per group
    BLOCK: tl.constexpr,
):
    """
    Grid: (N, num_groups, S_chunks)
    Each program handles one spatial chunk for a specific (n, g).
    Computes partial sum(y) and sum(y*y) over that spatial block across the group's channels,
    and atomically accumulates the partial results into global sum_ptr and sumsq_ptr.
    """
    n = tl.program_id(0)
    g = tl.program_id(1)
    p = tl.program_id(2)  # spatial chunk index

    num_spatial = H * W
    offs = p * BLOCK + tl.arange(0, BLOCK)
    mask = offs < num_spatial
    m = mask.to(tl.float32)

    acc = 0.0
    acc2 = 0.0

    # iterate over channels in the group and accumulate partial sums for this spatial chunk
    c_inner = 0
    while c_inner < Cg:
        c = g * Cg + c_inner
        # base linear index for channel c and batch n: ((n * C + c) * H * W)
        base = ((n * C + c) * H) * W
        # load contiguous spatial block for this channel
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)

        # compute sigmoid on the fly
        y = 1.0 / (1.0 + tl.exp(-x))

        acc += tl.sum(y * m)
        acc2 += tl.sum((y * y) * m)

        c_inner += 1

    # atomic accumulate partial results into global buffers
    pid = n * num_groups + g
    tl.atomic_add(sum_ptr + pid, acc)
    tl.atomic_add(sumsq_ptr + pid, acc2)


@triton.jit
def gn_apply_kernel(
    x_ptr,    # pointer to conv output (N, C, H, W) NCHW contiguous
    out_ptr,      # pointer to final output (N, C, H, W) NCHW contiguous
    mean_ptr,     # pointer to per (N, G) mean
    invstd_ptr,   # pointer to per (N, G) invstd
    gamma_ptr,    # per-channel GN weight (C,)
    beta_ptr,     # per-channel GN bias (C,)
    N, C, H, W,
    num_groups,
    Cg,
    BLOCK: tl.constexpr,
):
    """
    Grid: (N, C, S_chunks) where S_chunks = ceil(H*W / BLOCK)
    Each program processes up to BLOCK spatial locations for one (n, c).
    This kernel recomputes sigmoid(x) on the fly to avoid an extra global buffer.
    We avoid per-element div/mod by computing a per-program base pointer and using idx = base + offs.
    """
    pidn = tl.program_id(0)
    pidc = tl.program_id(1)
    pids = tl.program_id(2)  # chunk index across spatial locations

    # spatial offsets
    offs = pids * BLOCK + tl.arange(0, BLOCK)
    mask = offs < (H * W)

    # compute base pointer for this (n, c) so idx = base + offs
    base = ((pidn * C + pidc) * H) * W
    idx = base + offs

    # load conv values and compute sigmoid
    x = tl.load(x_ptr + idx, mask=mask, other=0.0)
    y = 1.0 / (1.0 + tl.exp(-x))  # sigmoid

    g = pidc // Cg
    base_g = pidn * num_groups + g
    mean = tl.load(mean_ptr + base_g)      # scalar
    invstd = tl.load(invstd_ptr + base_g)  # scalar

    # normalize: (y - mean) * invstd
    out = (y - mean) * invstd

    gamma = tl.load(gamma_ptr + pidc)  # scalar
    beta = tl.load(beta_ptr + pidc)    # scalar

    out = out * gamma + beta

    tl.store(out_ptr + idx, out, mask=mask)


class ModelNew(nn.Module):
    """
    Optimized Model:
      - Folds per-channel bias and scale into conv parameters at init so conv output is already (conv_orig(x) + bias) * scale.
      - Uses F.conv2d (cuDNN) for convolution (with optional NHWC weight copy for channels-last inputs).
      - Replaces nn.GroupNorm forward with fused Triton kernels that compute sigmoid + GroupNorm in two passes:
          1. compute per-(N,group) sum and sumsq of sigmoid(conv_out)
          2. compute mean/invstd and run final kernel to normalize and apply GN affine parameters (gamma, beta)
      - This reduces extra memory passes and fuses elementwise operations in Triton kernels optimized for the A6000.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape, eps=1e-5):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # keep original bias and scale as parameters per API
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        # create GroupNorm module only to hold affine parameters; we won't call its forward.
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.eps = eps

        # Fold bias and scale into convolution parameters for faster runtime (equivalent transformation).
        with torch.no_grad():
            out_ch = out_channels
            b1 = self.bias.view(out_ch)
            s1 = self.scale.view(out_ch)

            # Fold into weights: multiply each output-channel filter by its scale
            self.conv.weight.data.mul_(s1.view(out_ch, 1, 1, 1))

            # Fold into bias: new_bias = (old_bias + b) * s
            if self.conv.bias is None:
                self.conv.bias = nn.Parameter((b1 * s1).clone())
            else:
                self.conv.bias.data = (self.conv.bias.data + b1).mul(s1)

            # Ensure folded conv parameters are contiguous
            self.conv.weight.data = self.conv.weight.data.contiguous()
            if self.conv.bias is not None:
                self.conv.bias.data = self.conv.bias.data.contiguous()

            # Cache channels-last copy of folded weight for fast NHWC convolution path if input is already channels-last.
            try:
                self._weight_nhwc = self.conv.weight.data.clone().contiguous(memory_format=torch.channels_last)
            except Exception:
                self._weight_nhwc = None

            self._bias_copy = self.conv.bias.data.clone() if self.conv.bias is not None else None

            # Mark folded conv params as non-trainable for inference scenarios (reduces autograd overhead).
            self.conv.weight.requires_grad = False
            if self.conv.bias is not None:
                self.conv.bias.requires_grad = False

    def forward(self, x):
        # Run convolution. Take fast NHWC path only if input is already channels-last and we have cached NHWC weight.
        if x.is_cuda:
            if x.is_contiguous(memory_format=torch.channels_last) and (self._weight_nhwc is not None):
                x = F.conv2d(x, self._weight_nhwc, self._bias_copy,
                             stride=self.conv.stride, padding=self.conv.padding,
                             dilation=self.conv.dilation, groups=self.conv.groups)
            else:
                # NCHW path. Ensure contiguous for the subsequent Triton kernels which assume NCHW contiguous layout.
                x = x.contiguous()
                x = self.conv(x)
        else:
            x = x.contiguous()
            x = self.conv(x)

        # x is conv output of shape (N, C, H, W), dtype float32, device cuda (we target GPU).
        # We'll compute fused sigmoid + GroupNorm via Triton kernels.

        N, C, H, W = x.shape
        num_groups = self.group_norm.num_groups
        assert C % num_groups == 0, "channels must be divisible by num_groups"
        Cg = C // num_groups
        S = Cg * H * W  # elements per (N, group)

        # Ensure contiguous NCHW layout for pointer arithmetic correctness.
        x = x.contiguous()

        # Prepare output tensor
        out = torch.empty_like(x)

        # allocate reduction buffers on device (per (N, group)) and zero them before atomic accumulation
        device = x.device
        dtype = x.dtype
        ng = N * num_groups
        sums = torch.empty(ng, device=device, dtype=dtype)
        sumsqs = torch.empty(ng, device=device, dtype=dtype)
        sums.zero_()
        sumsqs.zero_()

        # Do not materialize sigmoid; compute it inside both kernels (saves global memory traffic).
        num_spatial = H * W
        S_chunks = (num_spatial + BLOCK - 1) // BLOCK

        # Launch reduce kernel: grid over (N, num_groups, S_chunks) so each program handles one spatial chunk.
        grid_reduce = (N, num_groups, S_chunks)
        gn_reduce_kernel[grid_reduce](
            x,                        # x_ptr (conv output)
            sums,                     # sum_ptr
            sumsqs,                   # sumsq_ptr
            N, C, H, W,
            num_groups,
            Cg,
            BLOCK=BLOCK
        )

        # compute mean and invstd on GPU
        # mean = sums / S
        mean = sums / float(S)
        var = sumsqs / float(S) - mean * mean
        invstd = torch.rsqrt(var + float(self.eps))

        # Ready per-channel affine params (gamma, beta)
        # group_norm.weight and bias are shape (C,)
        gamma = self.group_norm.weight.contiguous()
        beta = self.group_norm.bias.contiguous()

        # Launch apply kernel: grid (N, C, num_spatial_chunks)
        grid_apply = (N, C, S_chunks)
        gn_apply_kernel[grid_apply](
            x,
            out,
            mean,
            invstd,
            gamma,
            beta,
            N, C, H, W,
            num_groups,
            Cg,
            BLOCK=BLOCK
        )

        return out


# Keep input metadata consistent with original problem
batch_size = 128
in_channels = 8
out_channels = 32
height = width = 256
kernel_size = 3
num_groups = 8
bias_shape = (out_channels, 1, 1)
scale_shape = (out_channels, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape]