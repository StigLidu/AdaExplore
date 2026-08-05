import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configurations to find the best BLOCK_C, BLOCK_HW and warps/stages for Ampere (A6000)
# Prefer warp/vector aligned block sizes (multiples of 16/32) for better coalescing on Ampere.
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_C": 32,  "BLOCK_HW": 64},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_C": 64,  "BLOCK_HW": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_C": 128, "BLOCK_HW": 128}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_C": 64,  "BLOCK_HW": 256}, num_warps=8, num_stages=3),
]

# Tune over both spatial-block (BLOCK_HW), channel-block (BLOCK_C) and groups G
@triton.autotune(
    configs=AUTOTUNE_CONFIGS,
    key=["C", "H", "W", "G"],
)
@triton.jit
def _fused_logsumexp_hswish_kernel(
    x_conv_ptr,    # pointer to conv output [N, C, H, W]
    gamma_ptr,     # pointer to per-channel affine weight (gamma) [C]
    beta_ptr,      # pointer to per-channel affine bias (beta) [C]
    out_ptr,       # pointer to output [N, 1, H, W] flattened as N*HW
    N, C, H, W, G, eps,
    BLOCK_C: tl.constexpr, BLOCK_HW: tl.constexpr,
):
    """
    Spatial-blocked fused kernel with GroupNorm fused-in:
      - Each program handles BLOCK_HW contiguous spatial positions for a single batch n.
      - For each group (small number of channels), we:
         1) do a small reduction across group's channels to compute mean and variance per spatial lane
         2) re-iterate group's channels to normalize using the computed mean/var, apply per-channel
            affine gamma/beta, tanh->hardswish, add residual (conv) and update the online log-sum-exp.
      This avoids loading x_norm from memory and reduces global memory traffic.
    """
    HW = H * W
    nblocks = (HW + BLOCK_HW - 1) // BLOCK_HW

    pid = tl.program_id(0)  # one program per (n, hw_block)
    n = pid // nblocks
    block_idx = pid % nblocks
    hw_start = block_idx * BLOCK_HW

    ar_hw = tl.arange(0, BLOCK_HW)  # lane offsets within the spatial tile
    hw = hw_start + ar_hw
    mask_hw = hw < HW

    base_n = n * C * HW  # base pointer offset for this batch

    # very negative sentinel for masked lanes
    NEG_INF = -1e20

    # per-lane accumulators over the BLOCK_HW lanes for logsumexp
    cur_max = tl.full((BLOCK_HW,), NEG_INF, dtype=tl.float32)
    sumexp = tl.zeros((BLOCK_HW,), dtype=tl.float32)

    # channels per group (GroupNorm guarantees divisibility)
    Cg = C // G

    # iterate groups sequentially: compute per-group mean/var and immediately consume them
    for g in range(G):
        # small accumulators for the group's reduction (fp32)
        sum_g = tl.zeros((BLOCK_HW,), dtype=tl.float32)
        sumsq_g = tl.zeros((BLOCK_HW,), dtype=tl.float32)

        # first pass: compute sum and sumsq for this group across channels
        for k in range(Cg):
            ch = g * Cg + k
            # channel base offset and mask for spatial lanes
            ch_off = base_n + ch * HW
            mask = mask_hw  # group channels guaranteed in-range because Cg * G == C

            conv_vec = tl.load(x_conv_ptr + ch_off + hw, mask=mask, other=0.0)
            v = conv_vec  # inputs are float32 (we keep accumulations in fp32)
            sum_g += v
            sumsq_g += v * v

        # compute mean and invstd for the group (per spatial lane)
        mean = sum_g / Cg
        var = sumsq_g / Cg - mean * mean
        invstd = 1.0 / tl.sqrt(var + eps)

        # second pass: normalize each channel in the group, apply affine, activations, residual and logsumexp
        for k in range(Cg):
            ch = g * Cg + k
            ch_off = base_n + ch * HW
            mask = mask_hw

            # load conv again (we avoid loading x_norm by recomputing it)
            conv_vec = tl.load(x_conv_ptr + ch_off + hw, mask=mask, other=NEG_INF)
            v = conv_vec

            # normalize using group's mean/invstd
            normalized = (v - mean) * invstd

            # load per-channel affine parameters (scalars) and apply broadcasted affine
            gamma_val = tl.load(gamma_ptr + ch)
            beta_val = tl.load(beta_ptr + ch)
            y = normalized * gamma_val + beta_val

            # tanh via exp trick
            z = tl.exp(-2.0 * y)
            tanh_vec = (1.0 - z) / (1.0 + z)

            # HardSwish: x * clamp(x + 3, 0, 6) / 6
            tmp = tanh_vec + 3.0
            tmp = tl.maximum(tmp, 0.0)
            tmp = tl.minimum(tmp, 6.0)
            hsw_vec = tanh_vec * (tmp / 6.0)

            x_res = conv_vec + hsw_vec  # residual addition uses original conv

            # online stable log-sum-exp update per lane
            new_max = tl.maximum(cur_max, x_res)
            exp_term = tl.exp(x_res - new_max)
            sumexp = sumexp * tl.exp(cur_max - new_max) + exp_term
            cur_max = new_max

    out_val = cur_max + tl.log(sumexp)
    out_off = n * HW + hw
    tl.store(out_ptr + out_off, out_val, mask=mask_hw)


def triton_fused_logsumexp(x_conv: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, groups: int, eps: float):
    """
    Wrapper to launch the autotuned Triton kernel.
    Inputs:
      - x_conv: CUDA float32 contiguous tensor of shape [N, C, H, W]
      - gamma, beta: per-channel affine parameters (1D tensors of length C)
      - groups: number of groups for GroupNorm (must divide C)
      - eps: GroupNorm epsilon
    Returns tensor of shape [N, 1, H, W].
    """
    assert x_conv.is_cuda and gamma.is_cuda and beta.is_cuda, "Inputs and affine params must be CUDA tensors"
    assert x_conv.dtype == torch.float32 and gamma.dtype == torch.float32 and beta.dtype == torch.float32, "Only float32 supported for fused path"
    N, C, H, W = x_conv.shape
    assert gamma.numel() == C and beta.numel() == C, "gamma/beta must have one value per channel"
    assert C % groups == 0, "channels must be divisible by groups"

    x_conv_c = x_conv.contiguous()
    gamma_c = gamma.contiguous()
    beta_c = beta.contiguous()
    out = torch.empty((N, 1, H, W), device=x_conv_c.device, dtype=torch.float32)

    # flattened output pointer expects N*HW entries
    out_flat = out.view(N * H * W)

    # grid uses number of hw-blocks per batch (autotuner will supply BLOCK_HW via meta)
    grid = lambda meta: (N * ((H * W + meta['BLOCK_HW'] - 1) // meta['BLOCK_HW']),)

    # Launch autotuned kernel; decorator supplies BLOCK_C and BLOCK_HW as constexpr
    _fused_logsumexp_hswish_kernel[grid](x_conv_c, gamma_c, beta_c, out_flat, N, C, H, W, groups, eps)
    return out


class ModelNew(nn.Module):
    """
    Optimized Model that fuses Tanh, HardSwish, Residual Addition and LogSumExp reduction
    into a single high-performance Triton kernel for the channel-wise reduction.
    This keeps PyTorch's Conv2d and GroupNorm for correctness and uses Triton to
    minimize memory traffic and computation during the nonlinear + reduction stage.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(groups, out_channels, eps=eps)
        # keep PyTorch activations for API compatibility
        self.tanh = nn.Tanh()
        self.hard_swish = nn.Hardswish()

    def forward(self, x):
        x_conv = self.conv(x)
        # Fuse GroupNorm inside Triton kernel: pass affine params and group metadata
        gamma = self.group_norm.weight
        beta = self.group_norm.bias
        x_logsumexp = triton_fused_logsumexp(x_conv, gamma, beta, self.group_norm.num_groups, self.group_norm.eps)
        return x_logsumexp


# Keep original interface information
batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3
groups = 16

def get_inputs():
    # Provide CUDA float32 input for benchmarking/execution
    return [torch.rand(batch_size, in_channels, height, width).cuda().to(torch.float32)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, groups]