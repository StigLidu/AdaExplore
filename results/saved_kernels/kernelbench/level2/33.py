import torch
import torch.nn as nn
import triton
import triton.language as tl

# Triton kernel to compute per-block partial sums and sum-of-squares for each column tile.
# Input is expected in FP16; accumulations are done in FP32 for numeric stability.
@triton.jit
def _col_reduce_kernel(
    inp_ptr,              # pointer to input tensor (N, C) in FP16
    partial_sum_ptr,      # pointer to partial sums buffer (grid_m * C) in FP32
    partial_sumsq_ptr,    # pointer to partial sumsq buffer (grid_m * C) in FP32
    N,                    # rows
    C,                    # cols
    stride_row,           # stride between rows (in elements)
    stride_col,           # stride between cols (in elements)
    BLOCK_M: tl.constexpr,  # rows per block (constexpr)
    BLOCK_N: tl.constexpr,  # cols per block (constexpr)
):
    # block indices
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)

    row_start = row_block * BLOCK_M
    col_start = col_block * BLOCK_N

    rows = row_start + tl.arange(0, BLOCK_M)
    cols = col_start + tl.arange(0, BLOCK_N)

    row_mask = rows < N
    col_mask = cols < C
    mask = row_mask[:, None] & col_mask[None, :]

    offs = rows[:, None] * stride_row + cols[None, :] * stride_col
    ptrs = inp_ptr + offs

    # Load FP16 tile, cast to FP32 for accumulation
    x = tl.load(ptrs, mask=mask, other=0.0)
    x_fp32 = x.to(tl.float32)

    # Reduce across rows -> vector of length BLOCK_N (FP32)
    sum_cols = tl.sum(x_fp32, 0)
    sumsq_cols = tl.sum(x_fp32 * x_fp32, 0)

    # Store partials into flattened buffers at base index: base = row_block * C + cols
    base = row_block * C + cols
    tl.store(partial_sum_ptr + base, sum_cols, mask=col_mask)
    tl.store(partial_sumsq_ptr + base, sumsq_cols, mask=col_mask)


# Triton kernel to apply per-channel fused affine transform (y = x * eff_w + eff_b)
# Works in FP16 for the large activation matrix for bandwidth efficiency.
@triton.jit
def _apply_affine_kernel(
    inp_ptr,      # pointer to input (N, C) FP16
    out_ptr,      # pointer to output (N, C) FP16
    eff_w_ptr,    # pointer to effective weight (C,) FP16
    eff_b_ptr,    # pointer to effective bias (C,) FP16
    N,            # rows
    C,            # cols
    stride_row,   # stride between rows (elements)
    stride_col,   # stride between cols (elements)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)

    row_start = row_block * BLOCK_M
    col_start = col_block * BLOCK_N

    rows = row_start + tl.arange(0, BLOCK_M)
    cols = col_start + tl.arange(0, BLOCK_N)

    row_mask = rows < N
    col_mask = cols < C
    mask2 = row_mask[:, None] & col_mask[None, :]

    # Load per-channel fused params (1D)
    ew = tl.load(eff_w_ptr + cols, mask=col_mask, other=0.0)  # (BLOCK_N,)
    eb = tl.load(eff_b_ptr + cols, mask=col_mask, other=0.0)  # (BLOCK_N,)

    # Fast path when columns are contiguous (common row-major case)
    if stride_col == 1:
        offs = rows[:, None] * stride_row + cols[None, :]
        inp_ptrs = inp_ptr + offs
        out_ptrs = out_ptr + offs

        x = tl.load(inp_ptrs, mask=mask2, other=0.0)
        y = x * ew[None, :] + eb[None, :]
        tl.store(out_ptrs, y, mask=mask2)
        return

    # General case: arbitrary strides
    offs = rows[:, None] * stride_row + cols[None, :] * stride_col
    ptrs = inp_ptr + offs

    x = tl.load(ptrs, mask=mask2, other=0.0)
    y = x * ew[None, :] + eb[None, :]

    tl.store(out_ptr + offs, y, mask=mask2)


class ModelNew(nn.Module):
    """
    Optimized model that:
      - Uses PyTorch/cuBLAS for the GEMM under autocast to FP16.
      - Computes per-column mean/var using a Triton reduction kernel (FP16->FP32 accumulations).
      - Fuses scaling and BatchNorm affine application in a bandwidth-efficient Triton FP16 kernel.
    """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        # Keep original modules so parameter names and shapes remain compatible
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

        # Tunable Triton block sizes (constexpr). Balanced for Ampere A6000.
        # BLOCK_M controls number of rows reduced per block; larger BLOCK_M reduces number of partials.
        # BLOCK_N controls vector width for better memory throughput; keep it multiple of 32.
        self._BLOCK_M = 128
        self._BLOCK_N = 256

    def forward(self, x):
        # CPU fallback: keep original behavior
        if not x.is_cuda:
            x = self.gemm(x)
            x = x * self.scale
            return self.bn(x)

        device = x.device

        # GEMM under autocast to FP16 to reduce memory traffic
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            x_fp16 = self.gemm(x)  # (N, C) in FP16 due to autocast

        # Ensure contiguous layout for efficient Triton loads
        x_fp16 = x_fp16.contiguous()
        N, C = x_fp16.shape

        # Triton reduction parameters
        BLOCK_M = self._BLOCK_M
        BLOCK_N = self._BLOCK_N

        grid_m = (N + BLOCK_M - 1) // BLOCK_M
        grid_n = (C + BLOCK_N - 1) // BLOCK_N

        # Allocate partial buffers (grid_m x C) on device (FP32)
        # Flattened layout: partial_sum[block_row, col] stored at index block_row*C + col
        partial_sum = torch.zeros((grid_m, C), dtype=torch.float32, device=device)
        partial_sumsq = torch.zeros_like(partial_sum)

        # Launch Triton reduction kernel to compute per-block partial sums and sumsq
        _col_reduce_kernel[(grid_m, grid_n)](
            x_fp16, partial_sum, partial_sumsq,
            N, C, x_fp16.stride(0), x_fp16.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
        )

        # Final reduction across block rows to get per-column sums and sumsq (FP32)
        sum_per_col = partial_sum.sum(dim=0)      # (C,)
        sumsq_per_col = partial_sumsq.sum(dim=0)  # (C,)

        mean_x = sum_per_col / N
        var_x = sumsq_per_col / N - mean_x * mean_x
        var_x = var_x.clamp(min=0.0)

        # Account for the scale parameter: BatchNorm sees x * scale
        scale_fp32 = self.scale.to(torch.float32)

        if self.bn.training:
            mean_scaled = scale_fp32 * mean_x
            var_scaled = (scale_fp32 * scale_fp32) * var_x

            # Update running stats in FP32
            with torch.no_grad():
                m = self.bn.momentum if self.bn.momentum is not None else 0.1
                self.bn.running_mean.mul_(1 - m).add_(m * mean_scaled)
                self.bn.running_var.mul_(1 - m).add_(m * var_scaled)

            mean = mean_scaled
            var = var_scaled
        else:
            mean = self.bn.running_mean.to(torch.float32)
            var = self.bn.running_var.to(torch.float32)

        invstd = 1.0 / torch.sqrt(var + self.bn.eps)

        # BN affine params (gamma, beta) in FP32
        if self.bn.affine:
            gamma = self.bn.weight.to(torch.float32)
            beta = self.bn.bias.to(torch.float32)
        else:
            gamma = torch.ones(C, device=device, dtype=torch.float32)
            beta = torch.zeros(C, device=device, dtype=torch.float32)

        # Effective per-channel coefficients (FP32)
        # eff_w = scale * invstd * gamma
        # eff_b = -mean * invstd * gamma + beta
        eff_w_fp32 = (scale_fp32 * invstd) * gamma
        eff_b_fp32 = (-mean * invstd) * gamma + beta

        # Prepare FP16 inputs for the bandwidth-bound apply kernel
        inp_half = x_fp16.half()
        out_half = torch.empty_like(inp_half)

        # Convert coefficients to FP16 for the apply kernel (bandwidth heavy)
        eff_w = eff_w_fp32.contiguous().half()
        eff_b = eff_b_fp32.contiguous().half()

        # Grid and strides
        grid = (grid_m, grid_n)
        stride_row = inp_half.stride(0)
        stride_col = inp_half.stride(1)

        # Launch Triton kernel to apply fused affine in FP16
        _apply_affine_kernel[grid](
            inp_half, out_half,
            eff_w, eff_b,
            N, C, stride_row, stride_col,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
        )

        # Cast back to FP32 to match original model semantics
        out = out_half.to(torch.float32)
        return out