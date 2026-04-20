import torch
import torch.nn as nn
import triton
import triton.language as tl

# Enhanced autotune configs geared for NVIDIA A6000 (Ampere):
# Prefer warp-aligned BLOCK sizes (multiples of 32) and larger TILE_W values
# to improve throughput for the channel reduction on NHWC layout.
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 256, "TILE_W": 128}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK": 256, "TILE_W": 64},  num_warps=8, num_stages=4),
    triton.Config({"BLOCK": 128, "TILE_W": 128}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK": 128, "TILE_W": 64},  num_warps=8, num_stages=3),
    triton.Config({"BLOCK": 64,  "TILE_W": 64},  num_warps=8, num_stages=3),
    # Fallback smaller tiles for narrow W shapes
    triton.Config({"BLOCK": 64,  "TILE_W": 32},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 32,  "TILE_W": 32},  num_warps=4, num_stages=2),
]


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['C', 'W'])
@triton.jit
def _fused_act_res_logsumexp_kernel(
    x_conv_ptr,      # pointer to input x_conv (N, H, W, C) flattened (NHWC)
    x_norm_ptr,      # pointer to input x_norm (N, H, W, C) flattened (NHWC)
    out_ptr,         # pointer to output logsumexp (N, H, W), flattened
    N, C, H, W,      # sizes
    stride_n, stride_c, stride_h, stride_w,  # strides (in elements) for NHWC
    use_fp16,        # runtime flag (0/1) to enable mixed-precision elementwise path
    TILE_W: tl.constexpr,  # number of W positions handled by one program
    BLOCK: tl.constexpr,   # block size in channels
):
    """
    Fused kernel (NHWC layout expected):
      - computes tanh(x_norm) (optionally in fp16),
      - computes hardswish(tanh),
      - x_res = x_conv + hardswish,
      - computes numerically-stable logsumexp over channels for each (n,h,w)
    Each program handles one (n, h) and TILE_W contiguous w positions, iterating channels in chunks of BLOCK.
    Note: memory layout must be NHWC and pointers/strides must match NHWC.
    """
    pid = tl.program_id(0)

    # number of tiles along W
    num_tiles_w = (W + TILE_W - 1) // TILE_W
    hw_tiles = H * num_tiles_w

    # decode program id into n, h, tile
    n = pid // hw_tiles
    rem = pid - n * hw_tiles
    h = rem // num_tiles_w
    tile = rem - h * num_tiles_w

    w_start = tile * TILE_W

    # offsets for the TILE_W positions in W
    offs_w = tl.arange(0, TILE_W)
    w_idx = w_start + offs_w
    mask_w = w_idx < W  # which of the TILE_W positions are valid

    # base pointer for n and h at w_start (channels will be added per chunk)
    # NHWC flattened index: n*stride_n + h*stride_h + w*stride_w + c*stride_c
    base = n * stride_n + h * stride_h + w_start * stride_w

    offs_c = tl.arange(0, BLOCK)  # channel offsets within a chunk

    # running values per output position (length TILE_W)
    running_max = tl.full((TILE_W,), -1e30, dtype=tl.float32)
    running_sum = tl.zeros((TILE_W,), dtype=tl.float32)
    first = 1  # flag for initializing running values on first chunk

    inv6 = 1.0 / 6.0
    neg_inf = -1e30
    c = 0
    # iterate over channels in chunks of BLOCK
    while c < C:
        c_idx = c + offs_c  # vector of channel indices within this chunk
        mask_c = c_idx < C

        # compute addresses for loads: shape (BLOCK, TILE_W)
        # addrs = base + offs_w[None, :] * stride_w + c_idx[:, None] * stride_c
        addrs = base + offs_w[None, :] * stride_w + c_idx[:, None] * stride_c
        mask = mask_c[:, None] & mask_w[None, :]

        # Load x_conv and x_norm for this chunk. Supply safe other values so masked lanes
        # do not contribute to max (x_res_masked uses neg_inf).
        x_conv_vals = tl.load(x_conv_ptr + addrs, mask=mask, other=0.0)
        x_norm_vals = tl.load(x_norm_ptr + addrs, mask=mask, other=0.0)

        # Cast loaded inputs to fp32 for stable accumulation
        x_conv_fp32 = tl.cast(x_conv_vals, tl.float32)
        x_norm_fp32 = tl.cast(x_norm_vals, tl.float32)

        # Compute tanh. Optionally compute in fp16 to reduce compute bandwidth then cast back.
        if use_fp16 != 0:
            # clamp in fp16 to avoid overflow when exponentiating
            x_norm_fp16 = tl.cast(x_norm_fp32, tl.float16)
            xmax16 = tl.cast(20.0, tl.float16)
            x_clamped16 = tl.where(x_norm_fp16 > xmax16, xmax16, tl.where(x_norm_fp16 < -xmax16, -xmax16, x_norm_fp16))
            e2 = tl.exp(2.0 * x_clamped16)
            tanh_fp16 = (e2 - 1.0) / (e2 + 1.0)
            tanh_vals = tl.cast(tanh_fp16, tl.float32)
        else:
            xmax = 20.0
            x_clamped = tl.where(x_norm_fp32 > xmax, xmax, tl.where(x_norm_fp32 < -xmax, -xmax, x_norm_fp32))
            e2 = tl.exp(2.0 * x_clamped)
            tanh_vals = (e2 - 1.0) / (e2 + 1.0)

        # HardSwish on tanh_vals: hsw(x) = x * clamp(x + 3, 0, 6) / 6
        shifted = tanh_vals + 3.0
        shifted_clamped = tl.where(shifted < 0.0, 0.0, shifted)
        shifted_clamped = tl.where(shifted_clamped > 6.0, 6.0, shifted_clamped)
        hsw_vals = tanh_vals * (shifted_clamped * inv6)

        # x_res = x_conv + hsw (use fp32 conv)
        x_res_vals = x_conv_fp32 + hsw_vals

        # For invalid (out-of-range) channel positions, set to a large negative so they don't affect max/sum
        x_res_masked = tl.where(mask, x_res_vals, neg_inf)

        # compute chunk-wise max across the channel axis -> shape (TILE_W,)
        chunk_max = tl.max(x_res_masked, axis=0)

        # compute exp(vals - chunk_max) and sum across channel axis -> shape (TILE_W,)
        exp_vals = tl.exp(x_res_masked - chunk_max[None, :])
        chunk_sum = tl.sum(exp_vals, axis=0)

        if first == 1:
            running_max = chunk_max
            running_sum = chunk_sum
            first = 0
        else:
            # Numerically stable merge of partial logsumexp results:
            # new_max = max(running_max, chunk_max)
            # running_sum = running_sum * exp(running_max - new_max) + chunk_sum * exp(chunk_max - new_max)
            new_max = tl.maximum(running_max, chunk_max)
            running_sum = running_sum * tl.exp(running_max - new_max) + chunk_sum * tl.exp(chunk_max - new_max)
            running_max = new_max

        c += BLOCK

    # final logsumexp per position: m + log(s)
    out_vals = running_max + tl.log(running_sum + 1e-30)

    # compute flattened (N,H,W) base index and store results for the TILE_W positions
    pos_base = n * (H * W) + h * W
    out_addrs = pos_base + w_idx
    tl.store(out_ptr + out_addrs, out_vals, mask=mask_w)


def triton_fused_act_res_logsumexp(x_conv: torch.Tensor, x_norm: torch.Tensor):
    """
    Compute logsumexp over channel dimension after fusing tanh -> hardswish -> residual add:
      x_res = x_conv + hardswish(tanh(x_norm))
    Returns tensor of shape (N, H, W) containing logsumexp per spatial position.
    Inputs:
      x_conv : (N, C, H, W)
      x_norm : (N, C, H, W)
    """
    assert x_conv.is_cuda and x_norm.is_cuda, "Inputs must be on CUDA."
    assert x_conv.dtype == torch.float32 and x_norm.dtype == torch.float32, "Only float32 supported"
    # Ensure contiguous
    x_conv = x_conv.contiguous()
    x_norm = x_norm.contiguous()
    N, C, H, W = x_conv.shape

    # prepare output (N, H, W)
    out = torch.empty((N, H, W), device=x_conv.device, dtype=x_conv.dtype)

    # Flatten pointers for Triton
    x_conv_ptr = x_conv.reshape(-1)
    x_norm_ptr = x_norm.reshape(-1)
    out_ptr = out.reshape(-1)

    # compute strides for contiguous layout (N, C, H, W)
    stride_n = C * H * W
    stride_c = H * W
    stride_h = W
    stride_w = 1

    # grid: one program per (n, h, tile) where each tile covers TILE_W positions in W
    grid = lambda meta: (N * H * ((W + meta['TILE_W'] - 1) // meta['TILE_W']),)

    use_fp16 = 0  # inputs are fp32; keep fp32 elementwise computations for accuracy on this model
    _fused_act_res_logsumexp_kernel[grid](
        x_conv_ptr, x_norm_ptr, out_ptr,
        N, C, H, W,
        stride_n, stride_c, stride_h, stride_w,
        use_fp16,
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized model:
      - Keep PyTorch Conv2d and GroupNorm for correctness and parameter handling.
      - Fuse tanh -> hardswish -> residual addition and the channel-wise logsumexp reduction
        into a single high-throughput Triton kernel to avoid materializing large intermediates.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(groups, out_channels, eps=eps)
        # Keep activation modules for API compatibility (not used on the fused path)
        self.tanh = nn.Tanh()
        self.hard_swish = nn.Hardswish()

    def forward(self, x):
        # Convolution
        x_conv = self.conv(x)
        # Group Normalization
        x_norm = self.group_norm(x_conv)
        # Use fused Triton kernel to compute logsumexp across channels after computing activations on-the-fly
        logsumexp = triton_fused_act_res_logsumexp(x_conv, x_norm)  # shape (N, H, W)
        # keepdim to match original behavior (N, 1, H, W)
        x_logsumexp = logsumexp.unsqueeze(1)
        return x_logsumexp