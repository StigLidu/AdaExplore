import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune some reasonable block sizes for elementwise operations
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 256},  num_warps=2, num_stages=2),
    triton.Config({"BLOCK": 512},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=2),
]


@triton.autotune(
    configs=AUTOTUNE_CONFIGS,
    key=["n_elements"],
)
@triton.jit
def _relu_add_bias_kernel(
    x_ptr,          # pointer to input (flattened)
    bias_ptr,       # pointer to bias (C,)
    out_ptr,        # pointer to output (flattened)
    N,              # batch
    C,              # channels
    H,              # height
    W,              # width
    hw,             # H*W
    n_elements,     # total number of elements
    BLOCK: tl.constexpr,
):
    """
    For each linear index idx in [0, n_elements):
      c = (idx // hw) % C
      out[idx] = relu(x[idx]) + bias[c]
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK
    offsets = block_start + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    # Load input values (masked)
    x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute channel indices for each flattened offset:
    # idx // hw gives (n * C + c) for flattened by (N, C, H, W),
    # then % C yields channel c.
    # Note: hw and C are Python integers passed in as kernel args.
    c_idx = (offsets // hw) % C

    # Load bias per-channel (broadcast)
    bias_vals = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

    # ReLU implemented with where to avoid relying on tl.max
    relu_vals = tl.where(x_vals > 0.0, x_vals, 0.0)

    out_vals = relu_vals + bias_vals

    # Store result
    tl.store(out_ptr + offsets, out_vals, mask=mask)


def triton_relu_add_bias(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    x: tensor of shape (N, C, H, W), dtype float32, CUDA
    bias: tensor of shape (C, 1, 1) or (C,), dtype float32, CUDA
    Returns output tensor same shape and dtype as x.
    """
    assert x.is_cuda and bias.is_cuda, "Inputs must be CUDA tensors."
    assert x.dtype == torch.float32 and bias.dtype == torch.float32, "Only float32 supported."

    # Ensure contiguous
    x = x.contiguous()
    # Flatten bias to shape (C,)
    bias_flat = bias.view(-1).contiguous()

    N, C, H, W = x.shape
    hw = H * W
    n_elements = x.numel()

    out = torch.empty_like(x)

    # Flatten tensors for pointer arithmetic in the Triton kernel
    x_flat = x.view(-1)
    out_flat = out.view(-1)

    grid = lambda meta: ((n_elements + meta["BLOCK"] - 1) // meta["BLOCK"],)

    _relu_add_bias_kernel[grid](
        x_flat, bias_flat, out_flat,
        N, C, H, W, hw, n_elements
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized variant of the original Model:
      - Uses PyTorch's performant Conv2d for the convolution.
      - Fuses ReLU + per-channel bias addition using a custom Triton kernel.
    This keeps the convolution implementation (weights, bias) identical while
    accelerating the elementwise post-processing.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        # Keep the same Conv2d initialization as original model
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # Per-channel bias that will be added after ReLU; shape expected (C,1,1)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        # Convolution (uses PyTorch implementation)
        x = self.conv(x)
        # Fused ReLU + bias add via Triton kernel
        x = triton_relu_add_bias(x, self.bias)
        return x