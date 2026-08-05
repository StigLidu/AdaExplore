import torch
import torch.nn as nn
import triton
import triton.language as tl

# Triton kernel that fuses tanh -> scale -> bias -> maxpool for multiple output pool positions per program.
@triton.jit
def _fused_tanh_scale_bias_maxpool_kernel(
    inp_ptr,           # pointer to input tensor (conv output) (B, C, H_in, W_in) contiguous
    out_ptr,           # pointer to output tensor (B, C, H_pool, W_pool) contiguous
    B, C, H_in, W_in,  # ints
    H_pool, W_pool,    # ints
    pool_H, pool_W,    # ints (pool kernel size)
    stride,            # int (pool stride)
    scaling,           # float
    bias_ptr,          # pointer to bias tensor (C,1,1) contiguous
    BLOCK_P: tl.constexpr,  # number of output pooling positions processed by a program
):
    # grid: (B * C, num_blocks) where num_blocks = ceil(H_pool*W_pool / BLOCK_P)
    bc = tl.program_id(0)  # 0 .. B*C-1
    block_idx = tl.program_id(1)  # which block of pooling positions

    # decode batch and channel
    c = bc % C
    b = bc // C

    # start pooling position index for this block (linearized over H_pool*W_pool)
    p_start = block_idx * BLOCK_P
    offs = p_start + tl.arange(0, BLOCK_P)  # vector of pooling linear positions
    mask_p = offs < (H_pool * W_pool)

    # compute ph, pw for each pooled position
    ph = offs // W_pool
    pw = offs - ph * W_pool  # pw = offs % W_pool

    # input top-left coords for each pooling window
    in_h0 = ph * stride
    in_w0 = pw * stride

    # load bias for this channel and broadcast
    bias_val = tl.load(bias_ptr + c)
    bias_vec = bias_val + tl.zeros([BLOCK_P], dtype=tl.float32)  # broadcast

    # base index of the top-left element in the pooling window for each pooled position
    base = ((b * C + c) * H_in + in_h0) * W_in + in_w0  # vector of length BLOCK_P

    # initialize max to a very small number
    neg_inf = -1e30
    max_val = tl.full([BLOCK_P], neg_inf, dtype=tl.float32)

    # iterate over pooling window (small; unrolled by loops)
    h = 0
    while h < pool_H:
        w = 0
        # compute row offset for this h
        row_offset = h * W_in
        while w < pool_W:
            # addresses for all positions in this block for element (h,w)
            idx = base + row_offset + w
            # load input values (masked)
            v = tl.load(inp_ptr + idx, mask=mask_p, other=neg_inf)
            # apply tanh via stable formulation: tanh(x) = (1 - exp(-2x)) / (1 + exp(-2x))
            e = tl.exp(-2.0 * v)
            t = (1.0 - e) / (1.0 + e)
            s = t * scaling + bias_vec
            # elementwise max
            max_val = tl.where(s > max_val, s, max_val)
            w += 1
        h += 1

    # compute output linear indices and store results
    out_idx = ((b * C + c) * H_pool + ph) * W_pool + pw
    tl.store(out_ptr + out_idx, max_val, mask=mask_p)


def fused_tanh_scale_bias_maxpool(x: torch.Tensor, scaling: float, bias: torch.Tensor, pool_kernel_size: int):
    """
    x: conv output tensor with shape (B, C, H_in, W_in), contiguous, cuda, float32
    bias: tensor of shape (C,1,1) (or broadcastable), cuda, float32
    pool_kernel_size: int (assumed square)
    Returns fused result tensor of shape (B, C, H_pool, W_pool)
    """
    assert x.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    assert x.dtype == torch.float32 and bias.dtype == torch.float32

    x = x.contiguous()
    bias = bias.contiguous()

    B, C, H_in, W_in = x.shape
    pool_H = pool_W = pool_kernel_size
    stride = pool_kernel_size
    H_pool = (H_in - pool_H) // stride + 1
    W_pool = (W_in - pool_W) // stride + 1

    out = torch.empty((B, C, H_pool, W_pool), device=x.device, dtype=x.dtype, memory_format=torch.contiguous_format)

    # Choose a block size (number of pooled positions per Triton program).
    # This is tuned to provide a balance between throughput and occupancy on Ampere GPUs.
    # For pool_kernel_size small (4), a BLOCK_P of 128 performs well in practice.
    BLOCK_P = 128

    # Number of blocks along pooling positions dimension
    num_positions = H_pool * W_pool
    num_blocks = (num_positions + BLOCK_P - 1) // BLOCK_P

    # grid: (B * C, num_blocks)
    grid = (B * C, num_blocks)

    # Launch Triton kernel. Provide BLOCK_P as a constexpr.
    _fused_tanh_scale_bias_maxpool_kernel[grid](
        x, out,
        B, C, H_in, W_in,
        H_pool, W_pool,
        pool_H, pool_W,
        stride,
        float(scaling),
        bias,
        BLOCK_P=BLOCK_P,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model:
      - Uses native nn.Conv2d for convolution (kept for highly optimized cuDNN conv).
      - Fuses tanh -> scale -> bias -> maxpool into a single Triton kernel for the post-processing.
      - Kernel processes multiple pooled positions per Triton program to reduce launch overhead
        and improve memory throughput.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        # Keep PyTorch Conv2d for correctness and optimized convolution
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # store scaling factor
        self.scaling_factor = scaling_factor
        # bias parameter kept in same shape (C,1,1)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        # keep pool_kernel for metadata
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        # Convolution (kept as PyTorch op)
        x = self.conv(x)  # shape (B, C, H_in, W_in)
        # Fused tanh -> scale -> bias -> maxpool via Triton kernel
        x = fused_tanh_scale_bias_maxpool(x, self.scaling_factor, self.bias, self.pool_kernel_size)
        return x