import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configurations for block sizes (tuned for A6000-like devices)
# Added a small ROWS (rows per program) search dimension to improve occupancy
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 256, "ROWS": 2}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 512, "ROWS": 2}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 1024, "ROWS": 2}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK": 256, "ROWS": 4}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 512, "ROWS": 4}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK": 1024, "ROWS": 4}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['B', 'N'])
@triton.jit
def _softmax_rowwise_fused_kernel(
    x_ptr,        # pointer to input matrix (B x N)
    out_ptr,      # pointer to output matrix (B x N)
    B, N,         # matrix dimensions
    stride_b, stride_n,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
):
    """
    Compute row-wise softmax for an input matrix x (shape B x N).
    This kernel processes ROWS rows per program and performs:
      1) a streaming pass that computes row-wise (max, sum(exp(..-max))) using
         an online fusion to reduce global memory traffic,
      2) a second pass that writes normalized softmax outputs.
    Each Triton program handles ROWS rows (last program may have fewer; masks handle tails).
    """
    pid = tl.program_id(0)
    base_row = pid * ROWS
    row_offs = tl.arange(0, ROWS)                 # shape [ROWS]
    row_idx = base_row + row_offs                 # shape [ROWS]
    row_mask = row_idx < B                        # shape [ROWS]

    offs = tl.arange(0, BLOCK)                    # shape [BLOCK]

    # Streaming fused reduction: compute per-row running max and running sum (exp sums)
    running_max = tl.zeros((ROWS,), dtype=tl.float32) - 1e30
    running_sum = tl.zeros((ROWS,), dtype=tl.float32)

    start = 0
    while start < N:
        col_offs = start + offs                    # shape [BLOCK]
        col_mask = col_offs < N                    # shape [BLOCK]
        # ptrs shape [ROWS, BLOCK]
        ptrs = x_ptr + row_idx[:, None] * stride_b + col_offs[None, :] * stride_n
        mask = row_mask[:, None] & col_mask[None, :]
        # load with a safe other to compute block max
        vals = tl.load(ptrs, mask=mask, other=-1e30)    # [ROWS, BLOCK]
        block_max = tl.max(vals, axis=1)               # [ROWS]
        # compute sum of exp(vals - block_max) per row
        shifted = vals - block_max[:, None]
        s2 = tl.sum(tl.exp(shifted), axis=1)           # [ROWS]
        # online update for max and sum: keep numerical stability
        new_max = tl.maximum(running_max, block_max)
        # running_sum * exp(running_max - new_max) + s2
        running_sum = running_sum * tl.exp(running_max - new_max) + s2
        running_max = new_max
        start += BLOCK

    # Final pass: write normalized outputs
    inv_sum = 1.0 / running_sum
    start = 0
    while start < N:
        col_offs = start + offs
        col_mask = col_offs < N
        ptrs = x_ptr + row_idx[:, None] * stride_b + col_offs[None, :] * stride_n
        mask = row_mask[:, None] & col_mask[None, :]
        vals = tl.load(ptrs, mask=mask, other=0.0)
        out_vals = tl.exp(vals - running_max[:, None]) * inv_sum[:, None]
        out_ptrs = out_ptr + row_idx[:, None] * stride_b + col_offs[None, :] * stride_n
        tl.store(out_ptrs, out_vals, mask=mask)
        start += BLOCK


def triton_softmax_rowwise(x: torch.Tensor) -> torch.Tensor:
    """
    Compute softmax along last dimension (N) for a 2D tensor of shape (B, N)
    using the fused Triton row-wise kernel.
    """
    assert x.is_cuda, "Input must be on CUDA"
    assert x.dtype == torch.float32, "Input must be float32"
    assert x.dim() == 2, "Input must be 2D (B, N)"

    x_contig = x.contiguous()
    B, N = x_contig.shape
    out = torch.empty_like(x_contig)

    stride_b = x_contig.stride(0)
    stride_n = x_contig.stride(1)

    grid = lambda meta: ((B + meta['ROWS'] - 1) // meta['ROWS'],)

    _softmax_rowwise_fused_kernel[grid](
        x_contig, out, B, N, stride_b, stride_n
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized model that uses a Triton fused softmax kernel.
    The linear layer and dropout use PyTorch implementations; softmax is done in Triton.
    """
    def __init__(self, in_features, out_features, dropout_p):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        """
        x: (batch_size, in_features) float32 CUDA tensor
        """
        # Ensure input on CUDA
        if x.device.type != "cuda":
            x = x.cuda()

        x = self.matmul(x)
        x = self.dropout(x)
        # Triton softmax expects contiguous CUDA float32 2D tensor
        x = x.contiguous()
        x = triton_softmax_rowwise(x)
        return x


# Preserve original global constants for compatibility
batch_size = 128
in_features = 16384
out_features = 16384
dropout_p = 0.2


def get_inputs():
    # Return CUDA tensor to avoid implicit transfers during benchmarking
    return [torch.rand(batch_size, in_features, device="cuda")]


def get_init_inputs():
    return [in_features, out_features, dropout_p]