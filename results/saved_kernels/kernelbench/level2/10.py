import torch
import torch.nn as nn
import triton
import triton.language as tl


# Triton kernel: performs 2x2 max-pool (stride=2), hardtanh clamp, and partial sum reduction.
# Each program handles a tile (chunk) of pooled spatial locations for a specific (batch,channel) pair.
@triton.jit
def _pool_clamp_sum_kernel(
    inp_ptr,        # input pointer (N, C, H, W) contiguous
    out_ptr,        # output pointer (N*C,) where we accumulate partial sums via atomic add
    N, C, H, W,     # input dimensions
    H2, W2, H2W2,   # pooled dimensions and total pooled elements
    min_val, max_val,  # hardtanh bounds (float scalars)
    BLOCK: tl.constexpr,  # number of pooled elements processed per program
):
    # program indices
    block_idx = tl.program_id(0)   # which block of pooled positions
    idx_nc = tl.program_id(1)      # linear index for (n,c) pair: n*C + c

    # derive n and c from idx_nc
    n = idx_nc // C
    c = idx_nc - n * C

    # compute offsets for pooled positions this program will handle
    offs = block_idx * BLOCK + tl.arange(0, BLOCK)
    mask = offs < H2W2

    # pooled coords
    ph = offs // W2
    pw = offs - ph * W2  # offs % W2

    # corresponding input top-left coords for each pooled window (2x2, stride 2)
    h0 = ph * 2
    w0 = pw * 2

    # compute addresses for the four values in each 2x2 window
    # linear index for element ((n*C + c) * H + h) * W + w
    base_n_c = (n * C + c) * H * W
    idx00 = base_n_c + h0 * W + w0
    idx01 = base_n_c + h0 * W + (w0 + 1)
    idx10 = base_n_c + (h0 + 1) * W + w0
    idx11 = base_n_c + (h0 + 1) * W + (w0 + 1)

    # Load the four elements with masking. Use other=min_val to ensure masked lanes don't affect max
    a = tl.load(inp_ptr + idx00, mask=mask, other=min_val)
    b = tl.load(inp_ptr + idx01, mask=mask, other=min_val)
    c_ = tl.load(inp_ptr + idx10, mask=mask, other=min_val)
    d = tl.load(inp_ptr + idx11, mask=mask, other=min_val)

    # 2x2 max
    m1 = tl.maximum(a, b)
    m2 = tl.maximum(c_, d)
    m = tl.maximum(m1, m2)

    # hardtanh clamp
    m_clamped = tl.minimum(tl.maximum(m, min_val), max_val)

    # sum only the valid lanes
    m_sum = tl.sum(m_clamped * tl.cast(mask, tl.float32))

    # atomic add the partial sum to the output accumulator for this (n,c)
    # out_ptr is expected to be float32 contiguous of shape (N*C,)
    tl.atomic_add(out_ptr + idx_nc, m_sum)


def _fused_pool_clamp_mean(x: torch.Tensor, min_val: float, max_val: float):
    """
    x: input tensor of shape (N, C, H, W), contiguous on CUDA
    Performs 2x2 maxpool (stride=2), hardtanh clamp, and returns per-(N,C) mean over pooled spatial dims
    Returns tensor of shape (N, C)
    """
    assert x.is_cuda, "Input must be on CUDA"
    x = x.contiguous()
    N, C, H, W = x.shape
    # pooling parameters are fixed: kernel_size=2, stride=2
    assert H % 2 == 0 and W % 2 == 0, "H and W must be divisible by 2 for 2x2 pooling"
    H2 = H // 2
    W2 = W // 2
    H2W2 = H2 * W2

    # Prepare output tensor to accumulate sums (one scalar per (N,C))
    out = torch.zeros(N * C, device=x.device, dtype=x.dtype)

    # Triton kernel launch parameters
    BLOCK = 1024  # number of pooled elements per program (tunable)
    num_blocks = (H2W2 + BLOCK - 1) // BLOCK
    grid = lambda meta: (num_blocks, N * C)

    # Launch the kernel
    _pool_clamp_sum_kernel[grid](
        x,                  # inp_ptr
        out,                # out_ptr
        N, C, H, W,         # dims
        H2, W2, H2W2,       # pooled dims
        float(min_val),     # min_val
        float(max_val),     # max_val
        BLOCK=BLOCK
    )

    # compute mean by dividing by number of pooled elements
    out = out.view(N, C)
    mean = out / float(H2W2)
    return mean


class ModelNew(nn.Module):
    """
    Optimized model that uses a fused Triton kernel to replace MaxPool2d + Hardtanh + mean over spatial dims.
    The ConvTranspose2d remains as PyTorch implementation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        # Keep PyTorch ConvTranspose2d for correctness and performance
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        # store clamp bounds
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        # Note: maxpool params are assumed to be kernel=2, stride=2 for the fused kernel;
        # If different values are provided, fall back to PyTorch ops.
        self.maxpool_kernel_size = maxpool_kernel_size
        self.maxpool_stride = maxpool_stride

    def forward(self, x):
        x = self.conv_transpose(x)
        # If pooling is 2x2 stride 2, use fused Triton kernel
        if self.maxpool_kernel_size == 2 and self.maxpool_stride == 2 and x.is_cuda and x.dtype == torch.float32:
            # fused kernel computes per-(N,C) mean after pooling and clamping
            mean_nc = _fused_pool_clamp_mean(x, float(self.hardtanh_min), float(self.hardtanh_max))
            # apply tanh and reshape to (N, C, 1, 1)
            out = torch.tanh(mean_nc).unsqueeze(-1).unsqueeze(-1)
            return out
        else:
            # fallback to original sequence if parameters differ or not CUDA/fp32
            x = nn.functional.max_pool2d(x, kernel_size=self.maxpool_kernel_size, stride=self.maxpool_stride)
            x = torch.clamp(x, min=self.hardtanh_min, max=self.hardtanh_max)
            x = torch.mean(x, dim=(2, 3), keepdim=True)
            x = torch.tanh(x)
            return x


# Re-create the helper functions to match the original environment (input shapes and init args)
batch_size = 128
in_channels  = 64
out_channels = 64
height = width = 256
kernel_size  = 3
stride = 1
padding = 1
maxpool_kernel_size = 2
maxpool_stride = 2
hardtanh_min = -1
hardtanh_max = 1

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max]