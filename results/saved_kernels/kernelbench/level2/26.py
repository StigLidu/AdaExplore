import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configurations tuned for large elementwise workloads on Ampere GPUs
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 4096},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 8192},  num_warps=8, num_stages=2),
    triton.Config({"BLOCK_SIZE": 16384}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_SIZE": 32768}, num_warps=8, num_stages=3),
]

@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['n_elements'])
@triton.jit
def _fused_add_hardswish_kernel(out_ptr, x_ptr, add_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Out-of-place fused kernel:
      s = x + add
      out = s^2 * relu6(s + 3) / 6
    Writes result into out_ptr to avoid aliasing issues with in-place updates.
    Each program handles a contiguous block of size BLOCK_SIZE.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load original x and add values (masked loads)
    x_vals = tl.load(x_ptr + offs, mask=mask, other=0.0)
    add_vals = tl.load(add_ptr + offs, mask=mask, other=0.0)

    s = x_vals + add_vals
    # relu6(s + 3) = min(max(s + 3, 0), 6)
    relu6 = tl.minimum(tl.maximum(s + 3.0, 0.0), 6.0)
    out = s * s * (relu6 * (1.0 / 6.0))

    # Store result into out_ptr
    tl.store(out_ptr + offs, out, mask=mask)


def triton_fused_add_hardswish_inplace(x: torch.Tensor, add: torch.Tensor):
    """
    Wrapper for the Triton fused kernel.
    Allocates an output tensor and writes the fused result into it to avoid
    potential aliasing or correctness issues from in-place updates.
    Returns the new output tensor (does not mutate the original x).
    """
    assert x.is_cuda and add.is_cuda, "Tensors must be on CUDA."
    assert x.dtype == torch.float32 and add.dtype == torch.float32, "Only fp32 is supported."
    assert x.numel() == add.numel(), "Input tensors must have the same number of elements."

    # Ensure contiguous memory for predictable pointer arithmetic.
    x_contig = x if x.is_contiguous() else x.contiguous()
    add_contig = add if add.is_contiguous() else add.contiguous()

    # Allocate output tensor (same shape / dtype / device as x_contig)
    out = torch.empty_like(x_contig)

    n_elements = x_contig.numel()
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel which writes the result into out
    _fused_add_hardswish_kernel[grid](out, x_contig, add_contig, n_elements)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that keeps ConvTranspose3d in PyTorch (highly optimized)
    but fuses the subsequent elementwise addition and HardSwish-based activation
    into a single in-place Triton kernel to reduce memory traffic and allocations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        # Preserve the bias parameter (as in the original model signature)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x, add_input):
        """
        x: (batch_size, in_channels, D, H, W)
        add_input: (batch_size, out_channels, D*stride, H*stride, W*stride)
        Returns fused result (x is modified in-place inside the fused kernel).
        """
        x = self.conv_transpose(x)
        # Use the Triton fused in-place kernel to compute: out = (x + add_input) * hardswish(x + add_input)
        out = triton_fused_add_hardswish_inplace(x, add_input)
        return out