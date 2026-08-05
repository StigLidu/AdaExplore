import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

# Triton kernel that fuses channel-wise softmax (dim=1) with a MaxPool3d (cube) across spatial dims.
# For each output spatial position (n, d_out, h_out, w_out), the kernel:
#  1. Gathers the R = pool_k**3 input spatial positions that map to this output.
#  2. Computes softmax across channels for each input spatial position.
#  3. Takes the spatial max across those R softmax vectors to produce the pooled output vector (per channel).
# This avoids writing the full softmax tensor to memory and reading it again for pooling.
@triton.jit
def _softmax_maxpool3d_kernel(
    X_ptr, Y_ptr,
    N, C, D, H, W,         # input shape
    D_out, H_out, W_out,   # output spatial dims after pooling
    pool_k,                # pooling kernel size (cube)
    in_stride_N, in_stride_C, in_stride_D, in_stride_H, in_stride_W,
    out_stride_N, out_stride_C, out_stride_D, out_stride_H, out_stride_W,
    R: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)  # one program per output (n, d_out, h_out, w_out)
    M_out = N * D_out * H_out * W_out
    if pid >= M_out:
        return

    # decode output index -> (n, d_out, h_out, w_out)
    out_idx = pid
    S_out = D_out * H_out * W_out
    n = out_idx // S_out
    rem = out_idx % S_out
    d_out = rem // (H_out * W_out)
    rem2 = rem % (H_out * W_out)
    h_out = rem2 // W_out
    w_out = rem2 % W_out

    # Compute the R input spatial offsets inside the pooling cube
    r = tl.arange(0, R)  # shape (R,)
    # map r into local offsets within pool cube
    pk2 = pool_k * pool_k
    pd = r // pk2
    rem3 = r % pk2
    ph = rem3 // pool_k
    pw = rem3 % pool_k

    d_in = d_out * pool_k + pd
    h_in = h_out * pool_k + ph
    w_in = w_out * pool_k + pw

    # Base element offset (in elements) for each of the R input positions (channel=0)
    base = n * in_stride_N + d_in * in_stride_D + h_in * in_stride_H + w_in * in_stride_W  # (R,)

    col_idx = tl.arange(0, BLOCK)  # (BLOCK,)
    offs = base[:, None] + col_idx[None, :] * in_stride_C  # (R, BLOCK)
    mask_cols = col_idx < C  # (BLOCK,)

    # Some of the R positions may lie outside the input (when spatial dims are not divisible).
    # Create a row mask for valid spatial positions and combine with channel mask.
    valid_pos = (d_in < D) & (h_in < H) & (w_in < W)  # (R,)
    mask = valid_pos[:, None] & mask_cols[None, :]  # (R, BLOCK)

    # Load tile of shape (R, BLOCK), masked loads use very negative value for invalid entries
    x = tl.load(X_ptr + offs, mask=mask, other=-1e20)  # (R, BLOCK)

    # Per-input-position softmax across channels
    m = tl.max(x, axis=1)                      # (R,)
    ex = tl.exp(x - m[:, None])                # (R, BLOCK)
    s = tl.sum(ex, axis=1)                     # (R,)
    y_rows = ex / s[:, None]                   # (R, BLOCK)

    # Spatial max across the R positions -> result vector of length BLOCK (per channel)
    out_vals = tl.max(y_rows, axis=0)          # (BLOCK,)

    # Compute output base pointer for storing
    out_base = n * out_stride_N + d_out * out_stride_D + h_out * out_stride_H + w_out * out_stride_W
    out_offs = out_base + col_idx * out_stride_C  # (BLOCK,)
    tl.store(Y_ptr + out_offs, out_vals, mask=mask_cols)


def triton_softmax_maxpool3d(x: torch.Tensor, pool_k: int):
    """
    Fused softmax across channel (dim=1) and MaxPool3d with cubic kernel (pool_k x pool_k x pool_k).
    x: (N, C, D, H, W), CUDA float32 expected.
    Returns: y of shape (N, C, D_out, H_out, W_out)
    """
    if (not x.is_cuda) or x.dtype != torch.float32:
        # Fallback: compute softmax then pooling in PyTorch
        y = torch.softmax(x, dim=1)
        # apply single pooling with kernel=pool_k and stride=pool_k to emulate sequential pools fused
        return nn.functional.max_pool3d(y, kernel_size=pool_k, stride=pool_k)

    N, C, D, H, W = x.shape
    assert C <= 1024, "This kernel expects channels to be reasonably small."
    # Do not require exact divisibility of spatial dims by pool_k. PyTorch's pooling
    # semantics (kernel_size=K, stride=K) produce output size floor((L-K)/K)+1.
    # Use the same formula so we handle cases where conv reduces spatial dims (no padding).
    D_out = (D - pool_k) // pool_k + 1
    H_out = (H - pool_k) // pool_k + 1
    W_out = (W - pool_k) // pool_k + 1

    # Prepare output tensor (same dtype/device)
    y = torch.empty((N, C, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    # Kernel configuration
    BLOCK = int(C)        # set BLOCK to channel count (constexpr)
    R = int(pool_k ** 3)  # number of spatial positions per output (constexpr)

    # Ensure reasonable block size: if C is larger than a reasonable TL block, cap by splitting channels is possible,
    # but for our target architecture C=16 so it's small.
    assert BLOCK <= 1024

    # Fetch strides (in elements)
    sN, sC, sD, sH, sW = x.stride()
    oN, oC, oD, oH, oW = y.stride()

    # Grid: one program per output spatial location across batch
    M_out = N * D_out * H_out * W_out
    grid = ( (M_out + 1 - 1) // 1, )

    _softmax_maxpool3d_kernel[grid](
        x, y,
        N, C, D, H, W,
        D_out, H_out, W_out,
        pool_k,
        sN, sC, sD, sH, sW,
        oN, oC, oD, oH, oW,
        R=R, BLOCK=BLOCK,
        num_warps=4, num_stages=2
    )
    return y


class ModelNew(nn.Module):
    """
    Optimized model:
      - Uses PyTorch's Conv3d (cuDNN).
      - Fuses the channel-wise softmax and the two sequential MaxPool3d(pool_k) operations into
        a single Triton kernel that computes softmax per input spatial position and then takes
        the spatial max over the combined pooling cube (pool_k_combined = pool_k1 * pool_k2).
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        # we don't create separate pool layers; pooling is fused into the Triton kernel

        # store original pool kernel for reference (used to compute combined pooling)
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        x = self.conv(x)
        # Combine two sequential pools of size K into a single cube of size K*K
        combined_pool = self.pool_kernel_size * self.pool_kernel_size
        # Use Triton fused softmax + maxpool kernel
        x = triton_softmax_maxpool3d(x, combined_pool)
        return x


# Re-create the original helper variables and input functions (matching the original interface)
batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
pool_kernel_size = 2

def get_inputs():
    # Triton kernel expects CUDA fp32 tensors
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda().float()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, pool_kernel_size]