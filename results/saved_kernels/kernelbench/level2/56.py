import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Tuned block size for the A6000; larger block reduces loop overhead when
# reducing over a very large hidden dimension (32768). You can experiment
# with other values (2048, 8192) for further tuning.
BLOCK = 4096


@triton.jit
def _sigmoid_row_sum_kernel(
    y_ptr,        # pointer to input matrix (batch, n_cols)
    out_ptr,      # pointer to output vector (batch,)
    n_cols,       # number of columns (hidden_size)
    stride_row,   # stride between rows in elements
    stride_col,   # stride between columns in elements
    BLOCK: tl.constexpr,
):
    """
    Each program handles a single row. The kernel iterates over columns
    in blocks of size BLOCK, applies sigmoid, and accumulates the sum.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)  # offsets within the block
    row_ptr = y_ptr + row * stride_row  # pointer to the start of the row

    acc = 0.0  # accumulator for the row sum (scalar)

    # Iterate over column blocks
    for col_start in range(0, n_cols, BLOCK):
        cols = col_start + offs  # column indices for this block
        mask = cols < n_cols     # boolean mask for bounds

        # Compute addresses and load; use 'other=0.0' to zero-pad out-of-bounds
        vals = tl.load(row_ptr + cols * stride_col, mask=mask, other=0.0)

        # sigmoid: 1 / (1 + exp(-x))
        sig = 1.0 / (1.0 + tl.exp(-vals))

        # Accumulate masked sum. Use tl.where to zero out-of-bounds entries.
        acc += tl.sum(tl.where(mask, sig, 0.0))

    # Store the result
    tl.store(out_ptr + row, acc)


def triton_sigmoid_row_sum(y: torch.Tensor) -> torch.Tensor:
    """
    Fuse sigmoid + row-wise sum (dim=1) with a Triton kernel.
    Input:
      y: (batch, n_cols) float32 CUDA tensor
    Returns:
      (batch, 1) float32 CUDA tensor
    """
    assert y.is_cuda, "Triton kernel requires CUDA tensors."
    # ensure contiguous for simple stride logic
    y = y.contiguous()
    batch, n_cols = y.shape

    out = torch.empty(batch, dtype=y.dtype, device=y.device)

    # strides in number of elements (not bytes)
    stride_row = y.stride(0)
    stride_col = y.stride(1)

    # grid: one program per row
    grid = (batch,)

    # Launch Triton kernel. BLOCK is a constexpr parameter.
    _sigmoid_row_sum_kernel[grid](
        y, out, n_cols, stride_row, stride_col, BLOCK=BLOCK
    )

    return out.view(batch, 1)


class ModelNew(nn.Module):
    """
    Optimized model that keeps a standard Linear layer (to leverage cuBLAS
    for the heavy matmul) but fuses the subsequent sigmoid + row-sum into
    a single Triton kernel to reduce memory traffic and kernel launch overhead.
    """
    def __init__(self, input_size, hidden_size):
        super(ModelNew, self).__init__()
        # Keep same parameters for easy drop-in replacement
        self.linear = nn.Linear(input_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use PyTorch's optimized linear (cuBLAS) for matmul + bias
        y = F.linear(x, self.linear.weight, self.linear.bias)

        # If CUDA, use fused Triton kernel for sigmoid + sum across dim=1
        if y.is_cuda:
            return triton_sigmoid_row_sum(y)
        else:
            # Fallback to pure PyTorch on CPU
            return torch.sum(torch.sigmoid(y), dim=1, keepdim=True)


# Keep helper functions similar to original for compatibility
batch_size = 128
input_size = 32768
hidden_size = 32768


def get_inputs():
    return [torch.rand(batch_size, input_size)]


def get_init_inputs():
    return [input_size, hidden_size]