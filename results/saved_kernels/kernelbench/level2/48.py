import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configs tuned for A6000 / Ampere
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 4096}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK": 8192}, num_warps=8, num_stages=4),
]


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['N', 'C', 'S'])
@triton.jit
def fused_postops_kernel(
    inp_ptr,      # pointer to flattened input
    scale_ptr,    # pointer to per-channel scale (length C)
    bias_ptr,     # pointer to per-channel bias (length C)
    out_ptr,      # pointer to flattened output
    N,            # batch size
    C,            # number of channels
    S,            # spatial size (D*H*W)
    n_elements,   # total number of elements (kept for compatibility)
    BLOCK: tl.constexpr,
):
    # 2D grid: program_id(0) enumerates (n * C + c) entries, program_id(1) enumerates spatial tiles
    pid0 = tl.program_id(0)  # nc index (n * C + c)
    tid = tl.program_id(1)   # spatial tile index

    c = pid0 % C
    block_start = tid * BLOCK
    offs = block_start + tl.arange(0, BLOCK)
    mask = offs < S

    # compute global offsets for this (n,c) pair
    base = pid0 * S
    offs_global = base + offs

    # load input tile (vectorized)
    x = tl.load(inp_ptr + offs_global, mask=mask, other=0.0)

    # load per-channel scale and bias once (scalars) and broadcast
    s = tl.load(scale_ptr + c)
    b = tl.load(bias_ptr + c)

    # fused computation: y = tanh(x * s) * b, then sigmoid
    y = x * s
    e = tl.exp(y * 2.0)
    tanh_y = (e - 1.0) / (e + 1.0)
    z = tanh_y * b
    out = 1.0 / (1.0 + tl.exp(-z))

    tl.store(out_ptr + offs_global, out, mask=mask)


def triton_fused_postops(inp: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor, N: int, C: int, S: int):
    """
    inp: flattened contiguous input (fp32) on CUDA
    scale: 1D tensor length C on CUDA (fp32)
    bias: 1D tensor length C on CUDA (fp32)
    N, C, S: dimensions to build a 2D launch grid: grid = (N*C, ceildiv(S, BLOCK))
    """
    assert inp.is_cuda and scale.is_cuda and bias.is_cuda, "All tensors must be on CUDA."
    inp_contig = inp.contiguous()
    scale_contig = scale.contiguous()
    bias_contig = bias.contiguous()

    out = torch.empty_like(inp_contig)

    n_elements = inp_contig.numel()

    # grid launcher based on BLOCK constexpr: tile spatial dimension per (n,c)
    grid = lambda meta: (N * C, (S + meta['BLOCK'] - 1) // meta['BLOCK'])

    return out, grid, inp_contig, scale_contig, bias_contig, n_elements


class ModelNew(nn.Module):
    """
    Optimized Model:
      - Keep the highly-optimized conv3d in PyTorch (cuDNN)
      - Fuse the subsequent elementwise scaling, tanh, bias multiplication and sigmoid
        into a single Triton kernel that reads per-channel scale/bias (no expansion),
        minimizing memory traffic and improving throughput.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        # Keep same parameter shapes as original
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        # x: (N, C, D, H, W)
        x = self.conv(x)  # shape: (N, C, D, H, W)
        N, C, D, H, W = x.shape
        S = D * H * W
        device = x.device
        dtype = x.dtype

        # Prepare per-channel 1D scale and bias tensors of length C (no spatial expansion)
        scale_1d = self.scaling_factor.view(C).to(device=device, dtype=dtype)
        bias_1d = self.bias.view(C).to(device=device, dtype=dtype)

        # Flatten input
        inp_flat = x.contiguous().view(-1)

        # Prepare kernel wrapper outputs and grid (provide N, C, S for 2D tiling)
        out_buffer, grid, inp_contig, scale_contig, bias_contig, n_elements = triton_fused_postops(inp_flat, scale_1d, bias_1d, N, C, S)

        # Launch kernel with (N,C) x spatial tiles; pass N,C,S for kernel to compute offsets
        fused_postops_kernel[grid](inp_contig, scale_contig, bias_contig, out_buffer, N, C, S, n_elements)

        # reshape back to original conv output shape
        out = out_buffer.view(N, C, D, H, W)
        return out