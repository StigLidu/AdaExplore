import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune a few block sizes appropriate for Ampere GPUs.
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 128},  num_warps=2, num_stages=2),
    triton.Config({"BLOCK_SIZE": 256},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 512},  num_warps=8, num_stages=3),
]

@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['S', 'n_rows'])
@triton.jit
def _subtract_mean_kernel(
    inp_ptr,           # pointer to tensor (will be read and written in-place)
    S,                 # spatial size (D*H*W)
    n_rows,            # number of rows = N * C
    BLOCK_SIZE: tl.constexpr,
):
    """
    For each program (one per row) compute sum over S in a first loop, then subtract mean
    in a second loop, writing in-place to inp_ptr. This keeps work inside a single kernel
    invocation and avoids an extra large temporary tensor allocation in Python.
    """
    row = tl.program_id(0)  # each program handles one (N, C) row
    row_start = row * S

    # Phase 1: accumulate sum over spatial elements
    acc = 0.0
    num_iters = (S + BLOCK_SIZE - 1) // BLOCK_SIZE
    for i in range(num_iters):
        offs = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        idx = row_start + offs
        # mask elements that are within S
        mask = offs < S - i * BLOCK_SIZE
        vals = tl.load(inp_ptr + idx, mask=mask, other=0.0)
        acc += tl.sum(vals)

    # Compute mean for this row
    mean = acc / S

    # Phase 2: subtract mean and write back in-place
    for i in range(num_iters):
        offs = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        idx = row_start + offs
        mask = offs < S - i * BLOCK_SIZE
        vals = tl.load(inp_ptr + idx, mask=mask, other=0.0)
        out_vals = vals - mean
        tl.store(inp_ptr + idx, out_vals, mask=mask)


def triton_subtract_mean(x: torch.Tensor):
    """
    Subtract mean over spatial dims (D,H,W) for each (N, C) using a Triton kernel.
    Operates in-place on a contiguous copy (so the returned tensor is contiguous).
    """
    assert x.is_cuda and x.dtype == torch.float32, "Expect CUDA float32 tensor"
    x_contig = x.contiguous()
    N, C, D, H, W = x_contig.shape
    S = D * H * W
    n_rows = N * C

    # Grid: one program per (N, C) row
    grid = lambda meta: (n_rows,)

    # Launch Triton kernel (in-place)
    _subtract_mean_kernel[grid](x_contig, S, n_rows)
    return x_contig


class ModelNew(nn.Module):
    """
    Optimized model:
      - Use PyTorch's ConvTranspose3d and BatchNorm3d for correctness.
      - Subtract spatial mean using a fused Triton kernel for improved memory locality
        and reduced temporary allocation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                                 stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        # Subtract mean along spatial dims (2,3,4) using Triton kernel (returns contiguous tensor)
        x = triton_subtract_mean(x)
        return x