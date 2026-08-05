import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune several sensible tile sizes and warp/stage configs for A6000 (Ampere)
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_N": 64,  "BLOCK_C": 16}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_N": 128, "BLOCK_C": 16}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_N": 256, "BLOCK_C": 16}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_N": 512, "BLOCK_C": 16}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_N": 256, "BLOCK_C": 8},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_N": 128, "BLOCK_C": 8},  num_warps=4, num_stages=2),
]

@triton.autotune(AUTOTUNE_CONFIGS, key=["N", "C", "spatial"])
@triton.jit
def fused_softmax_sub_swish_max_kernel(
    x_ptr,        # pointer to flattened input (B*C*spatial)
    bias_ptr,     # pointer to bias (C,)
    out_ptr,      # pointer to flattened output (N,)
    N,            # total number of spatial positions = B * spatial
    spatial,      # D * H * W
    C,            # number of channels
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """
    Process a tile of BLOCK_N spatial positions and BLOCK_C channels.

    Memory layout assumptions:
      - x is flattened such that index for (b, c, p) is (b*C + c) * spatial + p
      - offs_n are global spatial positions [0 .. N-1] where N = B * spatial
    """
    pid = tl.program_id(0)
    start_n = pid * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)             # (BLOCK_N,)
    mask_pos = offs_n < N

    cidx = tl.arange(0, BLOCK_C)                         # (BLOCK_C,)
    mask_c = cidx < C

    # addresses computed as c*spatial + offs_n (works with flattened layout)
    addrs = cidx[:, None] * spatial + offs_n[None, :]    # (BLOCK_C, BLOCK_N)

    # Load values for this tile (BLOCK_C x BLOCK_N). Use a large negative for masked elems.
    vals = tl.load(x_ptr + addrs, mask=(mask_c[:, None] & mask_pos[None, :]), other=-1e20)

    # 1) Numerically stable softmax across channels for each spatial position
    max_per_pos = tl.max(vals, axis=0)                   # (BLOCK_N,)
    exps = tl.exp(vals - max_per_pos[None, :])           # (BLOCK_C, BLOCK_N)
    sum_exps = tl.sum(exps, axis=0)                      # (BLOCK_N,)
    softmax = exps / sum_exps[None, :]                   # (BLOCK_C, BLOCK_N)

    # 2) load per-channel bias once and subtract
    bias = tl.load(bias_ptr + cidx, mask=mask_c, other=0.0)   # (BLOCK_C,)
    tmp = softmax - bias[:, None]                         # (BLOCK_C, BLOCK_N)

    # 3) reduce max across channels of the tmp values
    tmp_max = tl.max(tmp, axis=0)                         # (BLOCK_N,)

    # 4) apply Swish: x * sigmoid(x)
    # sigmoid(x) = 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-tmp_max))
    swish = tmp_max * sig

    # write back results for valid positions
    tl.store(out_ptr + offs_n, swish, mask=mask_pos)


def triton_fused_softmax_sub_swish_max(x: torch.Tensor, bias: torch.Tensor):
    """
    Wrapper for the Triton fused kernel.
    x: (B, C, D, H, W) cuda fp32 contiguous tensor
    bias: (C,) cuda fp32 tensor
    returns: (B, D, H, W) cuda fp32 tensor
    """
    assert x.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    assert x.dtype == torch.float32 and bias.dtype == torch.float32

    # ensure contiguous for predictable flattening
    x = x.contiguous()
    bias = bias.contiguous()

    B, C, D, H, W = x.shape
    spatial = D * H * W
    N = B * spatial

    # Flatten x so that layout matches kernel's address calculation: index = (b*C + c) * spatial + p
    x_flat = x.view(-1)

    out = torch.empty((N,), device=x.device, dtype=x.dtype)

    # grid: one program handles BLOCK_N spatial positions
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)

    fused_softmax_sub_swish_max_kernel[grid](
        x_flat,
        bias,
        out,
        N,
        spatial,
        C,
    )

    out = out.view(B, D, H, W)
    return out


class ModelNew(nn.Module):
    """
    Optimized Model:
      - Uses PyTorch ConvTranspose3d and MaxPool3d for conv and pooling.
      - Fuses softmax over channels -> subtract per-channel bias -> swish -> channel-wise max
        into a single Triton kernel to minimize memory traffic and improve locality.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.max_pool = nn.MaxPool3d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        # Per-channel subtraction parameter
        self.subtract = nn.Parameter(torch.randn(out_channels, dtype=torch.float32))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.max_pool(x)
        # Fused Triton kernel performs: softmax across channels -> subtract per-channel bias -> swish -> max over channels
        x = triton_fused_softmax_sub_swish_max(x, self.subtract)
        return x