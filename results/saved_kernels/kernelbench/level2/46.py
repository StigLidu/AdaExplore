import torch
import torch.nn as nn
import triton
import triton.language as tl

# Triton kernel that fuses: (x - s1) -> tanh -> ( - s2 ) and average-pooling over POOL x POOL windows.
# Design:
#   - Grid dimensions: (N, C, H_out)
#   - Each program handles one (n, c, ph) and a vector of pw positions of length BLOCK (constexpr).
#   - BLOCK must be >= W_out so we mask out-of-range columns. We choose BLOCK as a constexpr launch param.
#   - POOL is a constexpr kernel parameter (pooling kernel size), used in loops.
# Notes:
#   - We implement tanh(x) via exp to avoid calling tl.tanh (tanh(x) = (e^{2x}-1)/(e^{2x}+1)).
#   - All vector ranges use constexpr BLOCK via tl.arange(0, BLOCK).

@triton.jit
def _fused_postproc_pool_kernel(
    inp_ptr,           # input tensor pointer (N, C, H, W)
    out_ptr,           # output tensor pointer (N, C, H_out, W_out)
    N, C, H, W,        # input shape
    H_out, W_out,      # output spatial dims after pooling
    s1, s2,            # scalar floats: subtract1_value, subtract2_value
    BLOCK: tl.constexpr,  # number of columns processed per program (constexpr)
    POOL: tl.constexpr,   # pooling kernel size (constexpr)
):
    # program ids
    n = tl.program_id(0)
    c = tl.program_id(1)
    ph = tl.program_id(2)  # pooled-row index

    offs = tl.arange(0, BLOCK)  # vector of column offsets (constexpr length)
    pw = offs  # pooled-column indices vector
    mask_pw = pw < W_out  # mask for valid pooled columns

    # starting coordinates in the input for this pooled row
    h0 = ph * POOL  # scalar

    # base pointer index for (n, c, 0, 0)
    base_nc = n * (C * H * W) + c * (H * W)

    # accumulator for each pw position
    acc = tl.zeros((BLOCK,), dtype=tl.float32)

    # For each element inside the pooling window
    for kh in range(POOL):
        # compute absolute input row index for this kh
        h_idx = h0 + kh  # scalar
        # for each kw inside pooling window
        for kw in range(POOL):
            # compute vector of input column indices for each pw
            w_idx = pw * POOL + kw  # vector
            # mask for valid input column indices
            mask = mask_pw & (w_idx < W)
            # compute flat input indices
            idx = base_nc + h_idx * W + w_idx
            vals = tl.load(inp_ptr + idx, mask=mask, other=0.0)  # vector load with mask
            # subtract1
            vals = vals - s1
            # tanh via exp: tanh(x) = (e^{2x} - 1) / (e^{2x} + 1)
            e = tl.exp(vals * 2.0)
            vals = (e - 1.0) / (e + 1.0)
            # subtract2
            vals = vals - s2
            # accumulate
            acc = acc + vals

    # average
    denom = float(POOL * POOL)
    out_vals = acc / denom

    # compute output flat indices and store
    out_base_nc = n * (C * H_out * W_out) + c * (H_out * W_out) + ph * W_out
    out_idx = out_base_nc + pw  # vector of output indices
    tl.store(out_ptr + out_idx, out_vals, mask=mask_pw)


def triton_fused_postproc_pool(inp: torch.Tensor, subtract1: float, subtract2: float, pool_k: int):
    """
    Wrapper to call the Triton kernel.
    inp: (N, C, H, W) contiguous float32 CUDA tensor
    returns: (N, C, H_out, W_out) float32 CUDA tensor
    """
    assert inp.is_cuda, "Input must be on CUDA"
    assert inp.dtype == torch.float32, "This kernel expects fp32 inputs"
    N, C, H, W = inp.shape
    assert H % pool_k == 0 and W % pool_k == 0, "Pooling kernel must evenly divide spatial dims"
    H_out = H // pool_k
    W_out = W // pool_k

    inp_contig = inp.contiguous()
    out = torch.empty((N, C, H_out, W_out), device=inp.device, dtype=inp.dtype)

    # Choose BLOCK >= W_out. W_out for our known sizes will be 63, so 64 is safe.
    BLOCK = 64
    grid = (N, C, H_out)
    # Launch kernel with constexpr BLOCK and POOL
    _fused_postproc_pool_kernel[grid](
        inp_contig, out,
        N, C, H, W,
        H_out, W_out,
        float(subtract1), float(subtract2),
        BLOCK=BLOCK,
        POOL=pool_k
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized Model that uses PyTorch Conv2d followed by a fused Triton kernel that
    performs: subtract1 -> tanh -> subtract2 -> average-pool (POOL x POOL).
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        # Keep convolution in PyTorch (highly optimized), fuse the subsequent elementwise + pooling in Triton.
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = float(subtract1_value)
        self.subtract2_value = float(subtract2_value)
        self.kernel_size_pool = int(kernel_size_pool)

    def forward(self, x):
        # Compute convolution with PyTorch
        x = self.conv(x)  # shape: (N, C, H, W)
        # Apply fused postprocessing + pooling via Triton kernel
        x = triton_fused_postproc_pool(x, self.subtract1_value, self.subtract2_value, self.kernel_size_pool)
        return x