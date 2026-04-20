import torch
import torch.nn as nn
import triton
import triton.language as tl

# Favor BLOCK sizes that align well on Ampere and give good occupancy.
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK": 512},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 256},  num_warps=4, num_stages=1),
    triton.Config({"BLOCK": 128},  num_warps=2, num_stages=1),
]


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['n_elements'])
@triton.jit
def _fused_elemwise_kernel(
    x_ptr,        # pointer to conv output
    orig_ptr,     # pointer to original (detached) conv output
    bias_ptr,     # pointer to compact per-channel bias (length C)
    out_ptr,      # pointer to output
    n_elements,   # total number of elements
    N, C, D, H, W,# shape integers for computing indices
    BLOCK: tl.constexpr,
):
    # Use a 2D grid: program_id(0) = channel, program_id(1) = linear block index that covers batches and spatial blocks.
    ch = tl.program_id(0)
    block_linear = tl.program_id(1)

    spatial = D * H * W
    blocks_per_batch = (spatial + BLOCK - 1) // BLOCK

    # Derive batch index and per-batch block index from the linear block index.
    n = block_linear // blocks_per_batch
    block_in_batch = block_linear % blocks_per_batch

    # Compute base flattened offset for this (n, ch, block_in_batch)
    base = ((n * C + ch) * spatial) + block_in_batch * BLOCK
    offs = base + tl.arange(0, BLOCK)
    mask = offs < n_elements

    # Load x and original values for this block
    x_vals = tl.load(x_ptr + offs, mask=mask, other=0.0)
    orig_vals = tl.load(orig_ptr + offs, mask=mask, other=0.0)

    # Load the single bias scalar for this channel once (scalar load).
    bias_val = tl.load(bias_ptr + ch)

    # fused computation: ((x + bias) + orig) * orig + orig
    tmp = x_vals + bias_val
    out = (tmp + orig_vals) * orig_vals + orig_vals

    tl.store(out_ptr + offs, out, mask=mask)


def fused_elementwise(x: torch.Tensor, orig: torch.Tensor, bias: torch.Tensor):
    # Ensure inputs are on CUDA and contiguous
    assert x.is_cuda and orig.is_cuda and bias.is_cuda, "All tensors must be on CUDA."
    x_c = x.contiguous()
    orig_c = orig.contiguous()
    # Expect bias to be compact (C,) or (C,1,1,1); make it a 1D contiguous tensor
    bias_c = bias.contiguous().view(-1)

    out = torch.empty_like(x_c)

    n_elements = x_c.numel()
    if n_elements == 0:
        return out

    # shape integers needed by the kernel
    N, C, D, H, W = x_c.shape
    spatial = D * H * W
    blocks_per_batch = (spatial + 1 - 1) // 1  # placeholder to ensure variable exists for lambda, real value uses meta below

    # grid for a 2D launch: (C, N * blocks_per_batch) where blocks_per_batch depends on meta["BLOCK"]
    grid = lambda meta: (C, N * ((spatial + meta["BLOCK"] - 1) // meta["BLOCK"]))

    # Launch kernel: pass compact bias and shape ints instead of an expanded bias tensor.
    _fused_elemwise_kernel[grid](x_c, orig_c, bias_c, out, n_elements, N, C, D, H, W)
    return out


class ModelNew(nn.Module):
    """
    Optimized Model: uses the original PyTorch ConvTranspose3d for correctness and
    a Triton kernel to fuse the subsequent elementwise operations:
      out = (((conv_out + bias) + orig) * orig) + orig

    Changes vs. the original:
    - Do not expand the bias to full tensor on the host; pass compact bias to the kernel.
    - Avoid an extra clone() copy of the conv output by using detach().
    - Use a Triton kernel that loads one bias scalar per block (per channel) and
      tiles across batches and spatial positions with a 2D grid for better locality.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        # Keep the ConvTranspose3d layer so parameters remain compatible
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        # Bias parameter as in the original model (shape typically (C,1,1,1))
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        # Run the convolution transpose as usual
        x = self.conv_transpose(x)

        # Keep a detached view of the original output (avoid an extra copy)
        original_x = x.detach()

        # Pass compact bias (flatten channel dimension) to the kernel instead of expanding.
        bias_compact = self.bias.contiguous().view(-1)

        # Use Triton fused kernel to compute: ((x + bias) + original_x) * original_x + original_x
        out = fused_elementwise(x, original_x, bias_compact)
        return out