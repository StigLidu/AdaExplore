import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configs to pick best BLOCK size for the reduction over M = H*W.
# Added very large BLOCK sizes (e.g., M itself) so the autotuner can pick a single-tile,
# single-load configuration when the per-row length M fits a single program tile.
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 256},   num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 512},   num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 1024},  num_warps=8, num_stages=3),
    triton.Config({"BLOCK": 2048},  num_warps=8, num_stages=3),
    triton.Config({"BLOCK": 4096},  num_warps=8, num_stages=4),
    triton.Config({"BLOCK": 8192},  num_warps=8, num_stages=4),
    triton.Config({"BLOCK": 16384}, num_warps=8, num_stages=4),  # M for 128*128
]

@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['M'])
@triton.jit
def _inst_norm_fused_kernel(
    x_ptr,              # pointer to input flattened (N*C, M) - expected FP32
    out_ptr,            # pointer to output flattened (N*C, M) - FP32 (may be same as x_ptr for in-place)
    N, C, H, W, M,      # dims
    recip,              # 1.0 / divide_by
    eps,                # epsilon for numerical stability
    BLOCK: tl.constexpr
):
    """
    FP32 path: supports both a two-pass reduction+apply (for tiled processing)
    and a single-pass load+reduce+apply when BLOCK == M (i.e., the whole row fits in one tile).
    This kernel writes FP32 outputs and can operate in-place if out_ptr == x_ptr.
    """
    row = tl.program_id(0)           # index in [0, N*C)
    offs = tl.arange(0, BLOCK)
    row_start = row * M

    # Fast single-tile path: if the chosen constexpr BLOCK exactly equals the row length M,
    # load the whole row once, compute stats, normalize, and store.
    if BLOCK == M:
        idx = row_start + offs
        mask = offs < M
        vals = tl.load(x_ptr + idx, mask=mask, other=0.0)  # FP32
        s = tl.sum(vals)
        sq = tl.sum(vals * vals)
        mean = s / M
        var = sq / M - mean * mean
        invstd = 1.0 / tl.sqrt(var + eps)
        out = (vals - mean) * invstd * recip
        tl.store(out_ptr + idx, out, mask=mask)
        return

    # General tiled two-pass path: first pass accumulates sum and sumsq
    s = 0.0
    sq = 0.0
    for start in range(0, M, BLOCK):
        idx = row_start + start + offs
        mask = (start + offs) < M
        vals = tl.load(x_ptr + idx, mask=mask, other=0.0)  # already FP32
        s += tl.sum(vals)
        sq += tl.sum(vals * vals)

    mean = s / M
    var = sq / M - mean * mean
    invstd = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and write (may overwrite the input buffer in-place)
    for start in range(0, M, BLOCK):
        idx = row_start + start + offs
        mask = (start + offs) < M
        vals = tl.load(x_ptr + idx, mask=mask, other=0.0)
        out = (vals - mean) * invstd * recip
        tl.store(out_ptr + idx, out, mask=mask)


# FP16 variant: load fp16, convert to fp32 for accumulation and math, write FP32 outputs.
# When using fp16 reads, we still write FP32 outputs into out_ptr (which should be a FP32 buffer).
@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['M'])
@triton.jit
def _inst_norm_fused_kernel_fp16(
    x_ptr,              # pointer to input flattened (N*C, M) - expected FP16
    out_ptr,            # pointer to output flattened (N*C, M) - FP32
    N, C, H, W, M,      # dims
    recip,              # 1.0 / divide_by
    eps,                # epsilon for numerical stability
    BLOCK: tl.constexpr
):
    """
    FP16 input path: supports single-tile (BLOCK == M) fast-path to avoid a second global load.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    row_start = row * M

    if BLOCK == M:
        idx = row_start + offs
        mask = offs < M
        vals_h = tl.load(x_ptr + idx, mask=mask, other=0.0)     # FP16 values loaded
        vals = vals_h.to(tl.float32)
        s = tl.sum(vals)
        sq = tl.sum(vals * vals)
        mean = s / M
        var = sq / M - mean * mean
        invstd = 1.0 / tl.sqrt(var + eps)
        out = (vals - mean) * invstd * recip
        tl.store(out_ptr + idx, out, mask=mask)
        return

    # Tiled two-pass
    s = 0.0
    sq = 0.0
    for start in range(0, M, BLOCK):
        idx = row_start + start + offs
        mask = (start + offs) < M
        vals_h = tl.load(x_ptr + idx, mask=mask, other=0.0)
        vals = vals_h.to(tl.float32)
        s += tl.sum(vals)
        sq += tl.sum(vals * vals)

    mean = s / M
    var = sq / M - mean * mean
    invstd = 1.0 / tl.sqrt(var + eps)

    for start in range(0, M, BLOCK):
        idx = row_start + start + offs
        mask = (start + offs) < M
        vals_h = tl.load(x_ptr + idx, mask=mask, other=0.0)
        vals = vals_h.to(tl.float32)
        out = (vals - mean) * invstd * recip
        tl.store(out_ptr + idx, out, mask=mask)


def triton_instance_norm_divide(x: torch.Tensor, divide_by: float, eps: float = 1e-5, use_fp16: bool = False):
    """
    Fused instance normalization (per N,C over H*W) followed by division by divide_by.

    Performance-oriented behavior:
      - FP32 path: operates in-place on the provided conv output buffer whenever possible
        (we pass the same buffer as both input and output). This eliminates an extra
        allocation and reduces memory traffic.
      - FP16 read path (optional): if use_fp16 is True, the function creates a contiguous
        FP16 view for input reads (halving read bandwidth) while writing FP32 outputs
        into the original FP32 conv buffer. This leaves the final output in FP32 while
        benefiting from reduced input bandwidth.
    """
    assert x.is_cuda, "Input must be on CUDA"

    # Ensure we have a contiguous FP32 buffer to write outputs into.
    x_contig = x.contiguous()
    N, C, H, W = x_contig.shape
    M = H * W
    n_ch = N * C
    recip = float(1.0 / divide_by)

    # single program per (N*C)
    grid = lambda meta: (n_ch,)

    if use_fp16:
        # Create an FP16 contiguous view for input reads to reduce input bandwidth.
        # Keep the FP32 contiguous buffer as the destination (in-place output).
        x_in_half = x_contig.half().contiguous()
        x_flat_in = x_in_half.view(n_ch, M)
        out_flat = x_contig.view(n_ch, M)  # FP32 destination (in-place)
        _inst_norm_fused_kernel_fp16[grid](
            x_flat_in, out_flat,
            N, C, H, W, M,
            recip, float(eps)
        )
        out = out_flat.view(N, C, H, W)
        return out
    else:
        # FP32 path: operate in-place by using the same buffer for input and output.
        x_flat = x_contig.view(n_ch, M)
        _inst_norm_fused_kernel[grid](
            x_flat, x_flat,   # input and output are the same buffer -> in-place
            N, C, H, W, M,
            recip, float(eps)
        )
        out = x_flat.view(N, C, H, W)
        return out


class ModelNew(nn.Module):
    """
    Optimized model:
      - Uses native torch.nn.Conv2d for convolution (keeps cuDNN)
      - Replaces InstanceNorm2d + division with a fused Triton kernel that computes
        per-(N,C) mean & invstd and applies normalization and division in one kernel launch.
      - The instance-norm/division is implemented to operate in-place on the conv output
        buffer (FP32 path) to reduce memory traffic. Optionally supports FP16 reads.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divide_by = divide_by
        self.eps = 1e-5

    def forward(self, x):
        x = self.conv(x)
        # Default: FP32 in-place normalization. Caller can opt-in to FP16 reads by
        # passing use_fp16=True to triton_instance_norm_divide if desired.
        x = triton_instance_norm_divide(x, self.divide_by, self.eps)
        return x


# Keep helper functions for input generation (matching the original API)
batch_size = 128
in_channels  = 64
out_channels = 128
height = width = 128
kernel_size = 3
divide_by = 2.0

def get_inputs():
    # Return CUDA tensor to ensure Triton kernel runs on device
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divide_by]