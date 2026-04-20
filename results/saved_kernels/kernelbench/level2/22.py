import torch
import torch.nn as nn
import triton
import triton.language as tl

# Triton kernel that:
#  - loads a row-major matrix A (BATCH x HIDDEN),
#  - applies clamp per element,
#  - computes row-wise logsumexp,
#  - computes Mish-like fused output final = x * mish(x) where x = logsumexp(row)
#    Note: mish(x) = x * tanh(softplus(x)) -> final = x * (x * tanh(softplus(x))) = x^2 * tanh(softplus(x))
# All of that computed inside the kernel to avoid extra host-device launches and intermediate allocations.
@triton.jit
def _logsumexp_clamp_mish_kernel(
    a_ptr,            # pointer to A data (fp32)
    out_ptr,          # pointer to output data (fp32) shape (BATCH,)
    BATCH,            # number of rows
    HIDDEN,           # number of columns
    a_row_stride,     # stride between rows
    a_col_stride,     # stride between columns
    clamp_min,        # float clamp min
    clamp_max,        # float clamp max
    BLOCK_H: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Which block of rows this program handles
    row_start = tl.program_id(0) * BLOCK_H
    rows = row_start + tl.arange(0, BLOCK_H)                # (BLOCK_H,)

    # column offsets for a chunk
    col_offsets = tl.arange(0, BLOCK_C)                     # (BLOCK_C,)

    # init per-row max to very small and sum to zero for the single-pass, numerically-stable update
    neg_inf = tl.full((BLOCK_H,), -1e20, dtype=tl.float32)  # (BLOCK_H,)
    max_vals = neg_inf
    sum_vals = tl.zeros((BLOCK_H,), dtype=tl.float32)

    # Single-pass: for each chunk compute chunk_max and chunk_sum (sum exp(vals - chunk_max))
    # then update per-row (max, sum) using a numerically-stable merge:
    #   if m > M: S = s + S * exp(M - m); M = m
    #   else:     S = S + s * exp(m - M)
    col_start = 0
    while col_start < HIDDEN:
        cols = col_start + col_offsets                         # (BLOCK_C,)
        mask_cols = cols < HIDDEN                              # (BLOCK_C,)

        # compute pointer matrix: rows[:,None] * a_row_stride + cols[None,:] * a_col_stride
        row_ptrs = rows[:, None] * a_row_stride                # (BLOCK_H, 1)
        col_ptrs = cols[None, :] * a_col_stride                # (1, BLOCK_C)
        ptrs = a_ptr + row_ptrs + col_ptrs                     # (BLOCK_H, BLOCK_C)

        # load values with masked "other" so invalid lanes don't affect computations
        vals = tl.load(ptrs, mask=mask_cols[None, :], other=clamp_min)  # (BLOCK_H, BLOCK_C)

        # apply clamping
        vals = tl.where(vals < clamp_min, clamp_min, vals)
        vals = tl.where(vals > clamp_max, clamp_max, vals)

        # compute per-row max for this chunk and sum of exp(vals - chunk_max)
        chunk_max = tl.max(vals, 1)                            # (BLOCK_H,)
        to_exp = vals - chunk_max[:, None]
        exp_vals = tl.exp(to_exp)
        chunk_sum = tl.sum(exp_vals, 1)                        # (BLOCK_H,)

        # numerically stable merge of (max_vals, sum_vals) with (chunk_max, chunk_sum)
        gt_mask = chunk_max > max_vals                        # (BLOCK_H,)

        # when chunk_max > max_vals:
        #   new_S = chunk_sum + sum_vals * exp(max_vals - chunk_max)
        # otherwise:
        #   new_S = sum_vals + chunk_sum * exp(chunk_max - max_vals)
        exp_m_diff = tl.exp(max_vals - chunk_max)              # exp(M - m)
        exp_c_diff = tl.exp(chunk_max - max_vals)             # exp(m - M)

        s_when_gt = chunk_sum + sum_vals * exp_m_diff
        s_when_le = sum_vals + chunk_sum * exp_c_diff

        sum_vals = tl.where(gt_mask, s_when_gt, s_when_le)
        max_vals = tl.where(gt_mask, chunk_max, max_vals)

        col_start += BLOCK_C

    # compute final logsumexp
    out_vals = max_vals + tl.log(sum_vals)  # (BLOCK_H,)

    # compute Mish fused: final = out_vals^2 * tanh(softplus(out_vals))
    # softplus(x) = log(1 + exp(x)); avoid log1p to be safe
    # use a numerically stable branch for large x
    large_mask = out_vals > 20.0
    exp_x = tl.exp(out_vals)
    soft_large = out_vals
    soft_small = tl.log(1.0 + exp_x)
    soft = tl.where(large_mask, soft_large, soft_small)  # (BLOCK_H,)

    # tanh(soft) computed via exp(-2*soft) to avoid using tl.tanh
    e = tl.exp(-2.0 * soft)
    tanh_sp = (1.0 - e) / (1.0 + e)

    final = out_vals * out_vals * tanh_sp  # x^2 * tanh(softplus(x))

    # store results for valid rows
    rows_mask = rows < BATCH
    tl.store(out_ptr + rows, final, mask=rows_mask)


class ModelNew(nn.Module):
    """
    Optimized Model using Triton to compute the reduction (logsumexp + Mish fusion) per row.
    The heavy matrix multiplication uses PyTorch's optimized Linear, while the subsequent
    clamp/logsumexp/mish are fused into a single Triton kernel to minimize memory traffic
    and kernel-launch overhead.
    """
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        # Keep the same linear layer for matmul; bias preserved
        self.matmul = nn.Linear(input_size, hidden_size)
        # Combine the scale and the "x = x + x" doubling into one scale factor
        self.scale_factor = float(scale_factor) * 2.0
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

    def forward(self, x):
        # x: (batch, input_size)
        assert x.is_cuda, "Input must be on CUDA"
        # 1) Linear
        x = self.matmul(x)  # (batch, hidden)

        # 2) in-place scale to avoid extra allocation
        x *= self.scale_factor

        # 3) fused kernel: clamp -> row-wise logsumexp -> mish fusion -> final
        batch, hidden = x.shape
        x = x.contiguous()
        out_flat = torch.empty(batch, device=x.device, dtype=x.dtype)

        # Tuned block sizes: BLOCK_H rows per program, BLOCK_C columns per chunk
        # For A6000 and these shapes, 64x256 is a good starting point.
        BLOCK_H = 64
        BLOCK_C = 256

        grid = ((batch + BLOCK_H - 1) // BLOCK_H,)

        _logsumexp_clamp_mish_kernel[grid](
            x,                       # a_ptr
            out_flat,                # out_ptr
            batch,                   # BATCH
            hidden,                  # HIDDEN
            x.stride(0),             # a_row_stride
            x.stride(1),             # a_col_stride
            self.clamp_min,          # clamp_min
            self.clamp_max,          # clamp_max
            BLOCK_H=BLOCK_H,
            BLOCK_C=BLOCK_C,
        )

        # reshape to (batch, 1) to match original model output shape
        return out_flat.view(batch, 1)


# Keep the helper interfaces similar to the original spec
batch_size = 1024
input_size = 8192
hidden_size = 8192
scale_factor = 2.0
clamp_min = -10.0
clamp_max = 10.0

def get_inputs():
    # ensure inputs are on GPU
    return [torch.rand(batch_size, input_size, device='cuda', dtype=torch.float32)]

def get_init_inputs():
    return [input_size, hidden_size, scale_factor, clamp_min, clamp_max]