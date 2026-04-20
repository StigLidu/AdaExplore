import torch
import torch.nn as nn
import triton
import triton.language as tl

# Triton kernel: for each block it computes averaged pooling over KxK windows,
# applies sigmoid, and atomically accumulates the per-block sums into an output
# array of size B (one scalar per batch element).
#
# The mapping:
#   For a given batch b, we consider all pooled positions and channels (C * H_pool * W_pool).
#   We flatten that into a 1D index "off" and compute:
#       c = off // (H_pool * W_pool)
#       tmp = off % (H_pool * W_pool)
#       ph = tmp // W_pool
#       pw = tmp % W_pool
#   The corresponding KxK input patch top-left coordinate is (ph * K, pw * K).
#   We load the KxK patch, compute its mean, apply sigmoid, and sum across offs in the block.
#
@triton.jit
def _pool_sig_sum_kernel(
    x_ptr,            # pointer to input tensor (flattened)
    out_partials_ptr, # pointer to per-program partial outputs (size = total programs)
    B, C, H, W,       # shapes of the input
    H_pool, W_pool,   # pooled spatial sizes
    n_elems,          # number of pooled outputs per batch (C * H_pool * W_pool)
    num_blocks,       # number of blocks per batch
    BLOCK: tl.constexpr,  # number of elements processed per program
    K: tl.constexpr       # pooling kernel size (constexpr)
):
    pid = tl.program_id(0)  # global program id
    b = pid // num_blocks   # batch index
    block_id = pid % num_blocks

    offs = block_id * BLOCK + tl.arange(0, BLOCK)  # offsets inside a batch's flattened pooled domain
    mask = offs < n_elems

    # Compute channel and pooled coordinates from flattened offset
    HWp = H_pool * W_pool
    c = offs // HWp
    tmp = offs % HWp
    ph = tmp // W_pool
    pw = tmp % W_pool

    # Compute base indices (flattened) for the top-left corner of each KxK pool patch
    # layout: N, C, H, W with row-major contiguous memory
    # index = b*(C*H*W) + c*(H*W) + (ph*K + kh)*W + (pw*K + kw)
    batch_offset = b * (C * H * W)
    channel_offset = c * (H * W)
    top_left = ph * (K * W) + pw * K  # ph*K*W + pw*K  (vectorized)
    base = batch_offset + channel_offset + top_left

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    # iterate over the KxK pool window; keep loads vectorized per lane and accumulate in registers
    for kh in range(0, K):
        row_offset = kh * W
        for kw in range(0, K):
            idxs = base + row_offset + kw
            vals = tl.load(x_ptr + idxs, mask=mask, other=0.0)
            acc += vals

    # compute mean over K*K using a precomputed reciprocal, apply sigmoid
    recip = 1.0 / float(K * K)
    mean = acc * recip
    sig = 1.0 / (1.0 + tl.exp(-mean))

    # Zero out invalid lanes and sum into a single per-program partial result
    sig_masked = tl.where(mask, sig, tl.zeros([BLOCK], dtype=tl.float32))
    block_sum = tl.sum(sig_masked)
    # Write the per-program partial result (no atomics, single store per program)
    tl.store(out_partials_ptr + pid, block_sum)


def triton_avgpool_sig_sum(x: torch.Tensor, pool_k: int):
    """
    x: tensor of shape (B, C, H, W), contiguous on CUDA
    pool_k: average pooling kernel size and stride
    returns: tensor of shape (B,) containing the sum over channels and pooled spatial dims
             of sigmoid(avg_pool(x))
    """
    assert x.is_cuda, "Input must be CUDA tensor."
    assert x.dtype == torch.float32, "This Triton kernel expects fp32 inputs."

    x = x.contiguous()
    B, C, H, W = x.shape
    K = int(pool_k)
    H_pool = H // K
    W_pool = W // K

    # number of pooled outputs per batch (C * H_pool * W_pool)
    n_elems = C * H_pool * W_pool
    if n_elems == 0:
        # degenerate case: return zeros
        return torch.zeros(B, device=x.device, dtype=x.dtype)

    # Choose block size; tradeoff between occupancy and per-program work.
    BLOCK = 256
    num_blocks = (n_elems + BLOCK - 1) // BLOCK
    total_programs = B * num_blocks
    grid = (total_programs,)

    # Allocate per-program partials buffer to avoid many small atomic adds
    out_partials = torch.zeros(total_programs, device=x.device, dtype=x.dtype)

    # Launch Triton kernel. Pass BLOCK and K as constexpr kwargs.
    _pool_sig_sum_kernel[grid](
        x, out_partials,
        B, C, H, W,
        H_pool, W_pool,
        n_elems, num_blocks,
        BLOCK=BLOCK, K=K
    )

    # Reduce per-program partials into final per-batch outputs.
    # Reshape to (B, num_blocks) and sum across the program dimension for each batch.
    out = out_partials.view(B, num_blocks).sum(dim=1)

    return out


class ModelNew(nn.Module):
    """
    Optimized model: keep the convolution as nn.Conv2d (leveraging highly-optimized cuDNN),
    and fuse AvgPool2d + Sigmoid + Sum into a custom Triton kernel for improved throughput.
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.pool_k = pool_kernel_size

    def forward(self, x):
        # x: (B, in_channels, H, W)
        x = self.conv(x)
        # Fuse: avg pool (kernel=pool_k, stride=pool_k), sigmoid, and sum across [1,2,3]
        # Implemented in Triton kernel for efficiency.
        x = triton_avgpool_sig_sum(x, self.pool_k)
        return x


# Keep the helper input functions for compatibility with the original harness.
batch_size = 128
in_channels = 8
out_channels = 64
height, width = 384, 384
kernel_size = 3
pool_kernel_size = 4

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda().float()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, pool_kernel_size]