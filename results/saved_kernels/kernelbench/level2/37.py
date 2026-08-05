import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch.cuda.amp import autocast

# Autotune configs tuned for very large N (e.g., 32768) on Ampere (A6000).
# We include a large BLOCK_N option (1024) to reduce launch overhead for huge batches,
# as well as smaller BLOCK_N/BLOCK_C options to allow autotuner flexibility.
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_N": 128,  "BLOCK_C": 64,  "STORE_INTERMEDIATE": False}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_N": 256,  "BLOCK_C": 64,  "STORE_INTERMEDIATE": True},  num_warps=8, num_stages=3),
    triton.Config({"BLOCK_N": 256,  "BLOCK_C": 128, "STORE_INTERMEDIATE": True},  num_warps=8, num_stages=3),
    triton.Config({"BLOCK_N": 512,  "BLOCK_C": 64,  "STORE_INTERMEDIATE": True},  num_warps=8, num_stages=3),
    triton.Config({"BLOCK_N": 512,  "BLOCK_C": 128, "STORE_INTERMEDIATE": True},  num_warps=8, num_stages=3),
    triton.Config({"BLOCK_N": 1024, "BLOCK_C": 64,  "STORE_INTERMEDIATE": True},  num_warps=8, num_stages=3),
]

@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['N', 'C', 'G', 'group_size'])
@triton.jit
def _fused_swish_bias_groupnorm_kernel(
    x_ptr,            # (N, C) FP16 input pointer
    bias_ptr,         # (C,) FP16 bias pointer
    gn_weight_ptr,    # (C,) FP16 groupnorm weight pointer
    gn_bias_ptr,      # (C,) FP16 groupnorm bias pointer
    out_fp32_ptr,     # (N, C) FP32 output pointer
    scratch_ptr,      # (N, C) FP16 scratch pointer used only if STORE_INTERMEDIATE else ignored
    N,                # batch size
    C,                # channels
    G,                # groups
    group_size,       # group size (C // G)
    group_size_f,     # float(group_size)
    eps,              # eps for groupnorm
    BLOCK_C: tl.constexpr,    # channels per tile
    BLOCK_N: tl.constexpr,    # rows per program
    STORE_INTERMEDIATE: tl.constexpr  # whether to store first-pass results to scratch_ptr
):
    """
    Fused kernel:
      v = Swish(x) + bias
      out = GroupNorm(v)  (with per-channel affine)
    Mixed precision:
      - inputs/params: FP16
      - accumulators/normalization: FP32
      - final outputs: FP32 (stored directly)
    Two-pass approach:
      1) accumulate per-row sum and sumsq across all channel tiles (FP32)
      2) recompute or reload transformed tiles to finalize normalized outputs
    """
    g = tl.program_id(0)  # group index
    nb = tl.program_id(1)  # batch-block index

    # Row indices handled by this program
    n_start = nb * BLOCK_N
    offs_n = n_start + tl.arange(0, BLOCK_N)                 # (BLOCK_N,)
    row_offs = offs_n[:, None] * C                           # (BLOCK_N, 1)

    # Channel tiling for this group
    c_base = g * group_size
    tiles = (group_size + BLOCK_C - 1) // BLOCK_C

    # Mask for valid rows
    mask_n = offs_n < N  # (BLOCK_N,)

    # FP32 accumulators per-row
    acc_sum = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc_sum_sq = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # First pass: compute Swish(x) + bias and accumulate sums. Optionally write to scratch.
    for t in range(tiles):
        offs_c = c_base + t * BLOCK_C + tl.arange(0, BLOCK_C)    # (BLOCK_C,)
        col_offs = offs_c[None, :]                               # (1, BLOCK_C)
        ptrs = x_ptr + row_offs + col_offs                       # (BLOCK_N, BLOCK_C)

        mask_c = (offs_c < C) & (offs_c < c_base + group_size)   # (BLOCK_C,)
        mask = mask_n[:, None] & mask_c[None, :]                 # (BLOCK_N, BLOCK_C)

        # Load input tile (FP16) -> cast to FP32
        x_tile = tl.load(ptrs, mask=mask, other=0.0)
        x_fp32 = tl.cast(x_tile, tl.float32)

        # Swish activation in FP32: x * sigmoid(x)
        sig = 1.0 / (1.0 + tl.exp(-x_fp32))
        v_fp32 = x_fp32 * sig

        # Add per-channel bias
        b_tile = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0)
        b_fp32 = tl.cast(b_tile, tl.float32)
        v_fp32 = v_fp32 + b_fp32[None, :]

        # accumulate sums and sum of squares
        acc_sum = acc_sum + tl.sum(v_fp32, axis=1)
        acc_sum_sq = acc_sum_sq + tl.sum(v_fp32 * v_fp32, axis=1)

        # Optionally store intermediate transformed tiles as FP16 to scratch to avoid recomputation
        if STORE_INTERMEDIATE:
            scratch_ptrs = scratch_ptr + row_offs + col_offs
            tl.store(scratch_ptrs, tl.cast(v_fp32, tl.float16), mask=mask)

    # finalize mean and invstd in FP32
    mean = acc_sum / group_size_f
    var = acc_sum_sq / group_size_f - mean * mean
    invstd = 1.0 / tl.sqrt(var + eps)

    # Second pass: reload transformed tile (if stored in scratch) or recompute, then normalize + affine
    for t in range(tiles):
        offs_c = c_base + t * BLOCK_C + tl.arange(0, BLOCK_C)
        col_offs = offs_c[None, :]
        ptrs = x_ptr + row_offs + col_offs
        out_ptrs = out_fp32_ptr + row_offs + col_offs

        mask_c = (offs_c < C) & (offs_c < c_base + group_size)
        mask = mask_n[:, None] & mask_c[None, :]

        if STORE_INTERMEDIATE:
            scratch_ptrs = scratch_ptr + row_offs + col_offs
            v_tile_fp16 = tl.load(scratch_ptrs, mask=mask, other=0.0)
            v_fp32 = tl.cast(v_tile_fp16, tl.float32)
        else:
            # recompute transformed tile
            x_tile = tl.load(ptrs, mask=mask, other=0.0)
            x_fp32 = tl.cast(x_tile, tl.float32)
            sig = 1.0 / (1.0 + tl.exp(-x_fp32))
            v_fp32 = x_fp32 * sig
            b_tile = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0)
            b_fp32 = tl.cast(b_tile, tl.float32)
            v_fp32 = v_fp32 + b_fp32[None, :]

        # load groupnorm affine params
        w_tile = tl.load(gn_weight_ptr + offs_c, mask=mask_c, other=0.0)
        gb_tile = tl.load(gn_bias_ptr + offs_c, mask=mask_c, other=0.0)
        w_fp32 = tl.cast(w_tile, tl.float32)
        gb_fp32 = tl.cast(gb_tile, tl.float32)

        # normalize and apply affine; out_fp32 stored directly as FP32
        out_fp32 = (v_fp32 - mean[:, None]) * invstd[:, None] * w_fp32[None, :] + gb_fp32[None, :]
        tl.store(out_ptrs, out_fp32, mask=mask)


def fused_swish_bias_groupnorm(x: torch.Tensor, bias_h: torch.Tensor, gn_weight_h: torch.Tensor, gn_bias_h: torch.Tensor, G: int, eps: float):
    """
    Wrapper for launching the autotuned Triton kernel.
    - x: (N, C) (CUDA) input tensor (will be converted to FP16 for the kernel).
    - bias_h, gn_weight_h, gn_bias_h: cached FP16 CUDA tensors of shape (C,).
    Returns FP32 tensor (kernel writes final results directly as FP32).
    """
    assert x.is_cuda, "x must be a CUDA tensor"
    assert bias_h is not None and gn_weight_h is not None and gn_bias_h is not None, "FP16 parameter copies must be provided"
    N, C = x.shape
    assert C % G == 0, "Channels must be divisible by groups"
    group_size = C // G
    group_size_f = float(group_size)

    # Ensure input is FP16 and contiguous for maximal memory throughput
    if x.dtype != torch.float16:
        x_in = x.half().contiguous()
    else:
        x_in = x.contiguous()

    # Final output as FP32 (avoid an extra device cast on the host)
    out_fp32 = torch.empty((N, C), device=x_in.device, dtype=torch.float32)

    # Allocate scratch FP16 buffer; used only by some autotuned configs.
    # Allocation is done here; large but necessary for the STORE_INTERMEDIATE variants.
    scratch = torch.empty_like(x_in)

    # Grid depends on chosen BLOCK_N in autotune meta
    grid = lambda meta: (G, (N + meta['BLOCK_N'] - 1) // meta['BLOCK_N'])

    _fused_swish_bias_groupnorm_kernel[grid](
        x_in, bias_h, gn_weight_h, gn_bias_h, out_fp32, scratch,
        N, C, G, group_size, group_size_f, float(eps)
    )
    return out_fp32


class ModelNew(nn.Module):
    """
    Optimized Model that:
      - Runs Linear under autocast(fp16) to leverage Tensor Cores.
      - Runs a fused Triton kernel that computes Swish + Bias + GroupNorm in two passes with FP32 accumulators.
      - Kernel writes final outputs as FP32 to avoid extra device-side casts.
    The Model caches FP16 device copies of bias and GroupNorm affine parameters to avoid repeated conversions.
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)

        # Keep canonical params in FP32
        self.bias.data = self.bias.data.float()
        self.group_norm.weight.data = self.group_norm.weight.data.float()
        self.group_norm.bias.data = self.group_norm.bias.data.float()

        # Cached FP16 device copies (created lazily)
        # register_buffer is used so they appear in state_dict but start as None
        self.register_buffer('bias_h', None)
        self.register_buffer('gn_weight_h', None)
        self.register_buffer('gn_bias_h', None)

    def _ensure_half_params(self, device):
        # Create or update cached FP16 parameter copies only when necessary
        if self.bias_h is None or self.bias_h.device != device or self.bias_h.shape != self.bias.shape:
            self.bias_h = self.bias.half().to(device).contiguous()
        if self.gn_weight_h is None or self.gn_weight_h.device != device or self.gn_weight_h.shape != self.group_norm.weight.shape:
            self.gn_weight_h = self.group_norm.weight.half().to(device).contiguous()
        if self.gn_bias_h is None or self.gn_bias_h.device != device or self.gn_bias_h.shape != self.group_norm.bias.shape:
            self.gn_bias_h = self.group_norm.bias.half().to(device).contiguous()

    def forward(self, x):
        # 1) Linear with mixed precision to leverage Tensor Cores
        with autocast(dtype=torch.float16):
            x_lin = self.matmul(x)

        # 2) Ensure cached FP16 params on the device and reuse them
        device = x_lin.device
        self._ensure_half_params(device)

        G = int(self.group_norm.num_groups)
        eps = float(self.group_norm.eps)

        # 3) Call fused Triton kernel (returns FP32 directly)
        out_fp32 = fused_swish_bias_groupnorm(x_lin, self.bias_h, self.gn_weight_h, self.gn_bias_h, G, eps)

        # 4) Return FP32 (preserves original model dtype semantics)
        return out_fp32