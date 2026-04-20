import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configurations to pick the best BLOCK (tile) and launch parameters for the device.
# These were chosen to cover a range of workloads for Ampere GPUs like the A6000.
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 128}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 256}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK": 512}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 512}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['M'])
@triton.jit
def _gelu_groupnorm_kernel(
    x_ptr,         # input tensor pointer (N*C*H*W,)
    y_ptr,         # temporary buffer pointer to store GELU(x) in fp16 (N*C*H*W,)
    gamma_ptr,     # per-channel scale (C,)
    beta_ptr,      # per-channel bias (C,)
    out_ptr,       # output tensor pointer (N*C*H*W,) fp32
    N, C, H, W, G, # tensor dims and number of groups
    eps,           # epsilon for stability (float)
    M,             # number of elements per group (channels_per_group * H * W)
    BLOCK: tl.constexpr,  # tile size (constexpr)
):
    # program handles one (n, g) pair
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    channels_per_group = C // G
    c_off = g * channels_per_group               # channel offset for this group
    hw_per_channel = H * W

    offs = tl.arange(0, BLOCK)                   # vector of indices in a tile

    # First pass: compute GELU(x), store in fp16 temp buffer y_ptr, accumulate sums in fp32
    s = 0.0
    ss = 0.0

    start = 0
    # iterate chunks covering M elements
    while start < M:
        idx = start + offs
        mask = idx < M

        # map flattened idx -> channel_local, hw
        c_local = idx // hw_per_channel
        hw = idx - c_local * hw_per_channel
        h = hw // W
        w = hw - h * W

        c_idx = c_off + c_local

        # linear index into NCHW flattened buffer: ((n*C + c_idx) * H + h) * W + w
        linear_idx = ((n * C + c_idx) * H + h) * W + w

        x = tl.load(x_ptr + linear_idx, mask=mask, other=0.0)

        # GELU (fp32), then cast to fp16 for storage
        y_fp32 = 0.5 * x * (1.0 + tl.erf(x / 1.4142135623730951))
        # store fp16 to temporary buffer to save memory bandwidth
        y_fp16 = tl.cast(y_fp32, tl.float16)
        tl.store(y_ptr + linear_idx, y_fp16, mask=mask)

        # accumulate in fp32 for numerical stability
        y_masked = tl.where(mask, y_fp32, 0.0)
        s = s + tl.sum(y_masked)
        ss = ss + tl.sum(y_masked * y_masked)

        start = start + BLOCK

    # compute mean and invstd for the group (fp32)
    mean = s / M
    var = ss / M - mean * mean
    invstd = 1.0 / tl.sqrt(var + eps)

    # Second pass: load precomputed GELU (fp16 -> fp32), normalize and apply affine (per-channel)
    start = 0
    while start < M:
        idx = start + offs
        mask = idx < M

        c_local = idx // hw_per_channel
        hw = idx - c_local * hw_per_channel
        h = hw // W
        w = hw - h * W

        c_idx = c_off + c_local

        linear_idx = ((n * C + c_idx) * H + h) * W + w

        # load precomputed GELU from temporary fp16 buffer and cast to fp32
        y_loaded_fp16 = tl.load(y_ptr + linear_idx, mask=mask, other=0.0)
        y = tl.cast(y_loaded_fp16, tl.float32)

        norm = (y - mean) * invstd

        gval = tl.load(gamma_ptr + c_idx, mask=mask, other=1.0)
        bval = tl.load(beta_ptr + c_idx, mask=mask, other=0.0)

        out = norm * gval + bval

        tl.store(out_ptr + linear_idx, out, mask=mask)

        start = start + BLOCK


class ModelNew(nn.Module):
    """
    Optimized model:
      - ConvTranspose2d is left to PyTorch (highly optimized).
      - GELU + GroupNorm are fused into a Triton kernel that computes GELU,
        reduces per-group mean/variance, normalizes and applies per-channel affine.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        # Keep GroupNorm to hold affine parameters and num_groups metadata
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        # If not on CUDA, fall back to reference implementation for correctness
        if not x.is_cuda:
            x = torch.nn.functional.gelu(x)
            x = self.group_norm(x)
            return x

        # Ensure contiguous for efficient Triton access
        x = x.contiguous()

        N, C, H, W = x.shape
        G = self.group_norm.num_groups
        assert C % G == 0, "Number of channels must be divisible by num_groups"

        # prepare gamma and beta (affine parameters). If they are None, create defaults.
        if self.group_norm.weight is None:
            gamma = torch.ones(C, dtype=x.dtype, device=x.device)
        else:
            gamma = self.group_norm.weight.contiguous()

        if self.group_norm.bias is None:
            beta = torch.zeros(C, dtype=x.dtype, device=x.device)
        else:
            beta = self.group_norm.bias.contiguous()

        # temporary buffer to hold GELU(x) in fp16 to reduce memory bandwidth
        y_fp16 = torch.empty_like(x, dtype=torch.float16)

        out = torch.empty_like(x)

        channels_per_group = C // G
        M = channels_per_group * H * W

        # Grid: one program per (n, group)
        grid = (N * G,)

        # eps for numerical stability
        eps = 1e-5

        # Launch the autotuned Triton kernel. The autotuner will pick an appropriate BLOCK.
        _gelu_groupnorm_kernel[grid](
            x, y_fp16, gamma, beta, out,
            N, C, H, W, G,
            eps,
            M
        )

        return out