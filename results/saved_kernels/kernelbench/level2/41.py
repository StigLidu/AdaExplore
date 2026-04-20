import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Reduction kernel: compute per-column sum and sum-of-squares across rows
@triton.jit
def _reduce_sum_sumsq_kernel(
    x_ptr,             # input tensor pointer (flattened)
    sums_ptr,          # per-column sums pointer (C,)
    sumsqs_ptr,        # per-column sum-of-squares pointer (C,)
    N,                 # number of rows (batch)
    C,                 # number of columns (features)
    BLOCK_R: tl.constexpr, # rows per inner load
    BLOCK_C: tl.constexpr  # columns per program
):
    col_block = tl.program_id(0)
    cols = col_block * BLOCK_C + tl.arange(0, BLOCK_C)      # (BLOCK_C,)
    mask_cols = cols < C

    # Accumulators for this column block
    sum_col = tl.zeros((BLOCK_C,), tl.float32)
    sumsq_col = tl.zeros((BLOCK_C,), tl.float32)

    row = 0
    # Iterate over row blocks to reduce across N
    while row < N:
        rows = row + tl.arange(0, BLOCK_R)                   # (BLOCK_R,)
        mask_rows = rows < N
        rows_i = rows[:, None]                               # (BLOCK_R, 1)
        cols_j = cols[None, :]                               # (1, BLOCK_C)
        offs = rows_i * C + cols_j                           # (BLOCK_R, BLOCK_C)
        mask2d = mask_rows[:, None] & mask_cols[None, :]
        x = tl.load(x_ptr + offs, mask=mask2d, other=0.0)    # (BLOCK_R, BLOCK_C)

        # Sum over rows dimension (axis 0)
        s = tl.sum(x, 0)
        ss = tl.sum(x * x, 0)
        sum_col += s
        sumsq_col += ss

        row += BLOCK_R

    # Store results for this column block
    tl.store(sums_ptr + cols, sum_col, mask=mask_cols)
    tl.store(sumsqs_ptr + cols, sumsq_col, mask=mask_cols)


# Fused kernel: per-feature affine (scale/shift), GELU (fast sigmoid approx), and ReLU
@triton.jit
def _fused_bn_gelu_relu_kernel(
    x_ptr,            # input tensor pointer (flattened)
    scale_ptr,        # per-channel affine scale pointer (C,)
    shift_ptr,        # per-channel affine shift pointer (C,)
    out_ptr,          # output tensor pointer (flattened)
    N,                # number of rows (batch)
    C,                # number of columns (features)
    BLOCK_R: tl.constexpr, # block size over rows
    BLOCK_C: tl.constexpr  # block size over columns
):
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)

    # 2D tile coordinates
    rows = row_block * BLOCK_R + tl.arange(0, BLOCK_R)    # (BLOCK_R,)
    cols = col_block * BLOCK_C + tl.arange(0, BLOCK_C)    # (BLOCK_C,)

    mask_rows = rows < N                                   # (BLOCK_R,)
    mask_cols = cols < C                                   # (BLOCK_C,)

    rows_i = rows[:, None]                                 # (BLOCK_R, 1)
    cols_j = cols[None, :]                                 # (1, BLOCK_C)

    # offsets into flattened (row-major) tensor -> (BLOCK_R, BLOCK_C)
    offs = rows_i * C + cols_j

    # Combined 2D mask for loads/stores
    mask2d = mask_rows[:, None] & mask_cols[None, :]

    # Load input tile (contiguous along columns for each row -> good coalescing)
    x = tl.load(x_ptr + offs, mask=mask2d, other=0.0)      # (BLOCK_R, BLOCK_C)

    # Load per-column scale/shift once per program and broadcast across rows
    scale = tl.load(scale_ptr + cols, mask=mask_cols, other=1.0)   # (BLOCK_C,)
    shift = tl.load(shift_ptr + cols, mask=mask_cols, other=0.0)   # (BLOCK_C,)

    # Apply affine transform using precomputed per-channel values
    y = x * scale[None, :] + shift[None, :]                 # broadcast across rows

    # GELU approximation: y * sigmoid(1.702 * y)
    s = 1.0 / (1.0 + tl.exp(-1.702 * y))
    y = y * s

    # ReLU
    y = tl.where(y > 0.0, y, 0.0)

    # Store back the tile
    tl.store(out_ptr + offs, y, mask=mask2d)


def triton_fused_bn_gelu_relu(x: torch.Tensor, bn: nn.BatchNorm1d):
    """
    Wrapper that launches a two-stage Triton pipeline:
      1) reduction kernel that computes per-column sums and sumsqs on-device
      2) finalize mean/var and compute per-channel affine scale/shift on-device
      3) elementwise kernel that applies affine + GELU + ReLU using the precomputed
         scale/shift.

    This removes the host-side torch.mean/torch.var passes and keeps all work
    on the GPU until the final elementwise pass.
    """
    assert x.is_cuda, "Input must be on CUDA."
    # Ensure contiguous row-major layout
    x = x.contiguous()
    N, C = x.shape
    device = x.device
    dtype = x.dtype

    # Prepare per-channel params (use defaults if None)
    gamma = bn.weight if bn.weight is not None else torch.ones(C, device=device, dtype=dtype)
    beta = bn.bias if bn.bias is not None else torch.zeros(C, device=device, dtype=dtype)
    eps = float(bn.eps)

    # Allocate device buffers for sums and sums of squares
    sums = torch.empty((C,), device=device, dtype=dtype)
    sumsqs = torch.empty((C,), device=device, dtype=dtype)

    # Reduction tile sizes (rows and columns)
    # Choose rows-chunk large to amortize overhead; columns chunk remains a multiple of 32.
    BLOCK_R = 64
    BLOCK_C = 128
    grid_reduce = ((C + BLOCK_C - 1) // BLOCK_C,)

    # Launch reduction kernel to compute per-column sum and sum-of-squares
    _reduce_sum_sumsq_kernel[grid_reduce](
        x, sums, sumsqs,
        N, C,
        BLOCK_R=BLOCK_R, BLOCK_C=BLOCK_C
    )

    # Finalize mean and var on-device (torch ops on CUDA tensors do not sync to host)
    rm = sums / float(N)
    rv = sumsqs / float(N) - rm * rm

    # Compute per-channel affine transform: scale and shift
    scale = gamma / torch.sqrt(rv + eps)
    shift = beta - rm * scale

    # Make sure parameter tensors are contiguous and on device
    scale = scale.contiguous()
    shift = shift.contiguous()

    out = torch.empty_like(x)

    # 2D tile sizes for the elementwise pass (rows per program, columns per program)
    # Using the same BLOCK_R/BLOCK_C values tuned above
    grid = ((N + BLOCK_R - 1) // BLOCK_R, (C + BLOCK_C - 1) // BLOCK_C)

    # Launch the elementwise fused kernel
    _fused_bn_gelu_relu_kernel[grid](
        x, scale, shift, out,
        N, C,
        BLOCK_R=BLOCK_R, BLOCK_C=BLOCK_C
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized Model that uses PyTorch for the Linear (GEMM) and a Triton kernel
    to fuse BatchNorm (using running stats), GELU (fast approx), and ReLU in one pass.
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        # Reuse the same Linear and BatchNorm interface as original model
        self.gemm = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features)
        # We will compute per-batch statistics in the fused wrapper (training-mode behavior)
        # so do not force the BatchNorm module into eval mode here.

    def forward(self, x):
        # GEMM using PyTorch's highly-optimized kernel
        x = self.gemm(x)
        # Fused BatchNorm (running stats) + GELU + ReLU via Triton
        x = triton_fused_bn_gelu_relu(x, self.batch_norm)
        return x