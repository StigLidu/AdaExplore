import torch
import torch.nn as nn
import triton
import triton.language as tl

# Triton kernel: fused elementwise add followed by "x * hardswish(x)" where hardswish(x) = x * relu6(x+3)/6
@triton.jit
def _add_then_hardswish_kernel(
    a_ptr,           # pointer to first input tensor (x after conv_transpose)
    b_ptr,           # pointer to second input tensor (add_input)
    out_ptr,         # pointer to output tensor
    n_elements,      # total number of elements
    BLOCK: tl.constexpr
):
    """
    Simpler 1-D indexing kernel: each program handles BLOCK contiguous elements.
    This avoids 2-D broadcasting/mask mismatches and is more robust.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK
    offsets = block_start + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)

    t = a + b
    # relu6(t + 3) = min(max(t + 3, 0), 6)
    relu6 = tl.minimum(t + 3.0, 6.0)
    relu6 = tl.maximum(relu6, 0.0)
    # hardswish(t) = t * relu6 / 6
    hsw = t * (relu6 / 6.0)
    out = t * hsw  # original model does x = x * hardswish(x)
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_add_then_hardswish(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor = None):
    """
    Wrapper to call the Triton kernel with a fixed, well-tested configuration.
    - No runtime JIT sweeping here (avoid many specializations).
    - Uses BLOCK elements per program (must be passed as a constexpr to the kernel).
    """
    assert a.is_cuda and b.is_cuda, "Inputs must be CUDA tensors"
    assert a.dtype == torch.float32 and b.dtype == torch.float32, "This kernel expects fp32 tensors"
    assert a.shape == b.shape, "Input shapes must match"

    a_contig = a.contiguous()
    b_contig = b.contiguous()

    # Prepare output buffer. If user provided `out` but it's not contiguous or not the same storage,
    # we'll use a contiguous temporary and copy back at the end.
    if out is None:
        out_buf = torch.empty_like(a_contig)
        need_copy_back = False
    else:
        assert out.is_cuda and out.shape == a_contig.shape and out.dtype == a_contig.dtype
        # If the provided out refers to the same storage as a_contig, we can operate on it directly.
        if out.data_ptr() == a_contig.data_ptr():
            out_buf = out
            need_copy_back = False
        else:
            out_buf = out.contiguous()
            # If out.contiguous() returned a different pointer than out (most cases), we'll copy back.
            need_copy_back = out_buf.data_ptr() != out.data_ptr()

    n_elements = a_contig.numel()

    # Fixed, well-tested configuration (constexpr passed to kernel). Avoid tuning in forward.
    BLOCK = 1024
    grid = ( (n_elements + BLOCK - 1) // BLOCK, )

    _add_then_hardswish_kernel[grid](a_contig, b_contig, out_buf, n_elements, BLOCK)

    if out is not None and need_copy_back:
        out.copy_(out_buf)

    return out_buf if out is None else out

class ModelNew(nn.Module):
    """
    Optimized model: uses PyTorch's ConvTranspose3d for the heavy convolution,
    and a fused Triton kernel for elementwise addition followed by the composite
    activation x * hardswish(x).
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        # Keep the bias parameter to match original signature (not used in forward to preserve original behavior)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x, add_input):
        x = self.conv_transpose(x)
        # Call the fused kernel in-place to avoid an extra allocation by default.
        # Use the optimized wrapper (no runtime tuning).
        return triton_add_then_hardswish(x, add_input, out=x)