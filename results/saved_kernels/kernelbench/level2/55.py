import torch
import torch.nn as nn
import triton
import triton.language as tl

# Two-phase Triton implementation:
# 1) _pool_partial_kernel: each program processes a contiguous block of pairs
#    (kernel_size=2 pairs) for a single batch row and writes a scalar partial
#    sum into a workspace buffer. No atomics are used because each (batch,block)
#    location is written by exactly one program.
# 2) _pool_reduce_kernel: each program reduces the partials for one batch row
#    by summing the partials and applying the scaling factor to produce the
#    final scalar output per batch element.
#
# This avoids the atomic-add overhead from many programs contending on the same
# output addresses and allows flexible tuning of BLOCK (pairs per program) and
# REDUCE_BLOCK (partials summed per loop iteration).
#
# This implementation expects kernel_size == 2 (pairwise max pooling). If a
# different kernel_size is requested, the Python fallback uses standard PyTorch ops.

@triton.jit
def _pool_partial_kernel(
    y_ptr,            # pointer to input y (batch, out_features)
    partial_ptr,      # pointer to partial workspace (batch * num_blocks)
    M,                # number of pairs (out_features // 2)
    stride_y,         # elements to advance to next row in y (y.stride(0))
    num_blocks,       # number of pair-blocks (for indexing into partials)
    BLOCK: tl.constexpr
):
    """
    Each program handles one (batch_idx, block_idx) and computes a scalar partial
    sum over up to BLOCK pairs (each pair is two adjacent features).
    """
    batch_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    # start pair index for this program
    p_start = block_idx * BLOCK
    offs = p_start + tl.arange(0, BLOCK)             # pair indices this program will handle
    mask = offs < M                                   # valid lanes

    # compute indices of the two members of each pair
    idx0 = 2 * offs
    idx1 = idx0 + 1

    # pointer to the start of this row
    row_ptr = y_ptr + batch_idx * stride_y

    # load values for both elements of each pair; use a large negative value for invalid lanes
    neg_inf = -1e30
    v0 = tl.load(row_ptr + idx0, mask=mask, other=neg_inf)
    v1 = tl.load(row_ptr + idx1, mask=mask, other=neg_inf)

    # pairwise max
    m = tl.maximum(v0, v1)

    # sum across lanes -> scalar partial
    partial = tl.sum(m, 0)

    # store partial at partial_ptr[batch_idx * num_blocks + block_idx]
    # compute flattened index
    dst_index = batch_idx * num_blocks + block_idx
    tl.store(partial_ptr + dst_index, partial)


@triton.jit
def _pool_reduce_kernel(
    partial_ptr,      # pointer to partial workspace (batch * num_blocks)
    out_ptr,          # pointer to final output (batch,)
    num_blocks,       # number of partials per batch row
    scale,            # float scalar multiplier
    REDUCE_BLOCK: tl.constexpr
):
    """
    Each program reduces partials for one batch row into a single scalar output.
    It loops over partials in chunks of size REDUCE_BLOCK using vector loads.
    """
    batch_idx = tl.program_id(0)
    row_ptr = partial_ptr + batch_idx * num_blocks

    acc = 0.0
    r_start = 0
    # loop over partials in chunks of size REDUCE_BLOCK
    while r_start < num_blocks:
        offs = tl.arange(0, REDUCE_BLOCK)
        mask = (r_start + offs) < num_blocks
        vals = tl.load(row_ptr + r_start + offs, mask=mask, other=0.0)
        acc = acc + tl.sum(vals, 0)
        r_start += REDUCE_BLOCK

    # store scaled final result
    tl.store(out_ptr + batch_idx, acc * scale)


def triton_pool_sum_scale_two_pass(y: torch.Tensor, scale: float, BLOCK: int = 2048, REDUCE_BLOCK: int = 1024):
    """
    Two-pass Triton pooling wrapper.

    Args:
        y: (batch, out_features) float32 CUDA tensor, contiguous.
        scale: scalar multiplier applied to final summed pooled value.
        BLOCK: number of pairs processed per program in the first pass (constexpr).
        REDUCE_BLOCK: chunk size for reduction in the second pass (constexpr).

    Returns:
        out: (batch,) float32 CUDA tensor.
    """
    assert y.is_cuda, "Input must be on CUDA."
    assert y.dtype == torch.float32, "This kernel expects fp32 inputs."

    y = y.contiguous()
    batch, out_features = y.shape
    num_pairs = out_features // 2  # MaxPool1d kernel_size=2 semantics

    # quick path
    if num_pairs == 0:
        return torch.zeros(batch, device=y.device, dtype=torch.float32)

    # number of blocks in pass 1
    num_blocks = (num_pairs + BLOCK - 1) // BLOCK

    # workspace for partials: flattened (batch * num_blocks)
    partial = torch.empty((batch * num_blocks,), device=y.device, dtype=torch.float32)

    # compute row stride in elements
    stride_y = y.stride(0)

    # Launch first kernel: grid (batch, num_blocks)
    grid1 = (batch, num_blocks)
    _pool_partial_kernel[grid1](
        y,                       # y_ptr
        partial,                 # partial_ptr
        num_pairs,               # M
        stride_y,                # stride_y
        num_blocks,              # num_blocks
        BLOCK=BLOCK
    )

    # Launch second kernel: one program per batch to reduce partials
    out = torch.empty((batch,), device=y.device, dtype=torch.float32)
    grid2 = (batch,)
    _pool_reduce_kernel[grid2](
        partial,                 # partial_ptr
        out,                     # out_ptr
        num_blocks,              # num_blocks
        float(scale),            # scale
        REDUCE_BLOCK=REDUCE_BLOCK
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized model that retains the PyTorch Linear (matmul via cuBLAS) and
    replaces MaxPool1d(kernel_size=2) + sum + scale with a fused two-pass
    Triton implementation that computes pairwise max and sums efficiently.

    Forward:
        x -> Linear(x) -> Triton two-pass pooling/reduce -> scaled outputs (batch,)
    """
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        # keep the Linear layer (uses cuBLAS/cuDNN)
        self.matmul = nn.Linear(in_features, out_features)
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor):
        # Perform linear layer (PyTorch optimized)
        y = self.matmul(x)

        # Ensure CUDA for Triton kernels
        if not y.is_cuda:
            y = y.cuda()

        # If kernel_size != 2, fallback to PyTorch ops to preserve semantics
        if self.kernel_size != 2:
            y_pooled = nn.functional.max_pool1d(y.unsqueeze(1), kernel_size=self.kernel_size).squeeze(1)
            out = torch.sum(y_pooled, dim=1) * self.scale_factor
            return out

        # Use the two-pass Triton fused pooling + sum + scale
        # Tune BLOCK and REDUCE_BLOCK for Ampere (A6000). These choices aim to
        # minimize the number of partials while keeping per-program resource use reasonable.
        out = triton_pool_sum_scale_two_pass(y, self.scale_factor, BLOCK=2048, REDUCE_BLOCK=1024)
        return out


# Preserve original helper API
batch_size = 128
in_features = 32768
out_features = 32768
kernel_size = 2
scale_factor = 0.5

def get_inputs():
    # Provide CUDA inputs for best performance
    return [torch.rand(batch_size, in_features).cuda().float()]

def get_init_inputs():
    return [in_features, out_features, kernel_size, scale_factor]