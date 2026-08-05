import torch
import torch.nn as nn
import triton
import triton.language as tl

# Triton kernel: fused elementwise operations
# For each element x:
# out = 2 * x * x + (bias[c] + 1) * x
@triton.jit
def fused_elemwise_kernel(
    x_ptr,         # pointer to input tensor (N, C, D, H, W) flattened
    bias_ptr,      # pointer to bias per channel, shape (C,)
    out_ptr,       # pointer to output tensor (same shape as input)
    N,             # batch size
    C,             # channels
    spatial,       # D * H * W
    stride_n,      # stride between batches in elements (C * spatial)
    stride_c,      # stride between channels in elements (spatial)
    BLOCK: tl.constexpr,  # block size (constexpr)
):
    # program ids: pid_nc indexes (n * C + c), pid_block indexes blocks along spatial dim
    pid_nc = tl.program_id(0)
    pid_block = tl.program_id(1)

    c = pid_nc % C
    n = pid_nc // C

    block_start = pid_block * BLOCK
    offs = block_start + tl.arange(0, BLOCK)
    mask = offs < spatial

    # compute the flat indices for this (n, c) channel slice
    index = n * stride_n + c * stride_c + offs

    x = tl.load(x_ptr + index, mask=mask, other=0.0)
    b = tl.load(bias_ptr + c)

    out = 2.0 * x * x + (b + 1.0) * x

    tl.store(out_ptr + index, out, mask=mask)


def triton_fused_elementwise(x: torch.Tensor, bias: torch.Tensor, BLOCK=1024):
    """
    x: Tensor of shape (N, C, D, H, W), contiguous CUDA float32
    bias: Tensor of shape (C, 1, 1, 1) or (C,), contiguous CUDA float32
    Returns fused output tensor.
    """
    assert x.is_cuda and bias.is_cuda, "Input tensors must be on CUDA"
    assert x.dtype == torch.float32 and bias.dtype == torch.float32, "Only fp32 supported"
    x = x.contiguous()
    # flatten bias to 1D per channel
    bias_flat = bias.contiguous().view(-1)

    N, C, D, H, W = x.shape
    spatial = D * H * W
    out = torch.empty_like(x)

    stride_w = 1
    stride_h = W
    stride_d = H * W
    stride_c = spatial
    stride_n = C * spatial

    # grid: first dim iterate over (n * C), second over blocks in spatial dim
    num_blocks = (spatial + BLOCK - 1) // BLOCK
    grid = (N * C, num_blocks)

    fused_elemwise_kernel[grid](
        x, bias_flat, out,
        N, C, spatial,
        stride_n, stride_c,
        BLOCK
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized Model that keeps the ConvTranspose3d in PyTorch and fuses the
    subsequent elementwise operations into a single Triton kernel.
    The fused operation computes:
        out = 2 * X * X + (bias + 1) * X
    where X is the output of the conv_transpose layer.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        # bias used in the fused elementwise op, shape (C,1,1,1)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        # Compute conv transpose with PyTorch (keeps highly optimized conv implementation)
        x = self.conv_transpose(x)
        # Fuse the sequence of elementwise ops into one Triton kernel
        out = triton_fused_elementwise(x, self.bias)
        return out