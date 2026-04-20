import torch
import torch.nn as nn
import triton
import triton.language as tl

# Triton kernel: fuse ReLU -> GELU(approx) -> Sigmoid -> add bias
# GELU is approximated as x * sigmoid(1.702 * x) for efficiency and to avoid tanh/erf.
@triton.jit
def _fused_acts_bias_kernel(
    x_ptr,        # pointer to input/output tensor (flattened)
    bias_ptr,     # pointer to bias values (length C)
    out_ptr,      # pointer to output tensor (flattened)
    n_elements,   # total number of elements
    C,            # number of channels
    plane,        # spatial plane size = D * H * W
    BLOCK: tl.constexpr,  # block size (constexpr)
):
    # 3D grid: (plane_blocks, C, N)
    pid_plane = tl.program_id(0)  # which block along the spatial plane
    pid_c = tl.program_id(1)      # channel index
    pid_n = tl.program_id(2)      # batch index

    # start offset within the spatial plane for this block
    block_start_in_plane = pid_plane * BLOCK
    offs = tl.arange(0, BLOCK)
    # mask lanes that go beyond the plane size in the last block
    remain = plane - block_start_in_plane
    mask = offs < remain

    # base global offset for (n, c, plane_block)
    # layout flatten: ((n * C + c) * plane) + offset_in_plane
    base = (pid_n * C + pid_c) * plane + block_start_in_plane
    global_offs = base + offs

    # load contiguous spatial elements for this (n, c) pair
    x = tl.load(x_ptr + global_offs, mask=mask, other=0.0)

    # load scalar bias for this channel (safe because grid second dim is C)
    b = tl.load(bias_ptr + pid_c, mask=pid_c < C, other=0.0)

    # ReLU
    x = tl.where(x > 0.0, x, 0.0)

    # GELU approximation: x * sigmoid(1.702 * x)
    a = 1.702 * x
    sig_a = 1.0 / (1.0 + tl.exp(-a))
    gel = x * sig_a

    # Final sigmoid: sigmoid(gel)
    out = 1.0 / (1.0 + tl.exp(-gel))

    # Add scalar bias (broadcasted across the block)
    out = out + b

    # Store result (contiguous stores across the spatial plane)
    tl.store(out_ptr + global_offs, out, mask=mask)


def triton_fused_acts_bias(x: torch.Tensor, bias: torch.Tensor):
    """
    Wrapper to launch the Triton fused kernel.
    x: tensor of shape [N, C, D, H, W], cuda, contiguous
    bias: tensor of shape [C] or [C,1,1,1], cuda, contiguous
    """
    assert x.is_cuda and bias.is_cuda, "Inputs must be on CUDA"
    x = x.contiguous()
    # Ensure bias is 1D contiguous vector of length C
    if bias.dim() != 1:
        bias_vec = bias.view(-1).contiguous()
    else:
        bias_vec = bias.contiguous()

    out = torch.empty_like(x)
    N, C, D, H, W = x.shape
    plane = D * H * W
    n_elements = x.numel()

    # Tuned block size for Ampere: smaller than 1024 to reduce register/shared pressure
    BLOCK = 256

    # Number of blocks along the spatial plane
    plane_blocks = (plane + BLOCK - 1) // BLOCK

    # 3D grid: (plane_blocks, C, N) so each program handles a contiguous block of the plane for a single (n,c)
    grid = lambda meta: (plane_blocks, C, N)

    # Launch kernel: pass tensors directly; provide BLOCK as constexpr
    _fused_acts_bias_kernel[grid](
        x,
        bias_vec,
        out,
        n_elements,
        C,
        plane,
        BLOCK=BLOCK,
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized model that uses the same Conv3d layer but fuses the subsequent
    activations and bias addition into a single Triton kernel for better throughput.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        # Keep the convolution as PyTorch native (highly optimized)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        # Additional bias parameter as in the original model (shape: [C,1,1,1])
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        # Run convolution using PyTorch
        x = self.conv(x)
        # Apply fused activations + bias via Triton kernel
        # Ensure bias is proper shape/contiguity on the same device
        bias_vec = self.bias.to(x.device)
        return triton_fused_acts_bias(x, bias_vec)