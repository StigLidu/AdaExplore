import torch
import torch.nn as nn
import triton
import triton.language as tl

# Triton kernel implementing fused LayerNorm (across channels) + GELU + scaling.
# This kernel assumes the input is of shape (Nrows, C) in row-major contiguous layout,
# where each row corresponds to a vector over channels to be normalized.
# It computes: out = scale * GELU( (x - mean) / sqrt(var + eps) * weight + bias )
# where weight and bias are per-channel parameters (length C).
@triton.jit
def _fused_ln_gelu_scale_kernel(
    x_ptr,          # pointer to input, shape (Nrows, C) row-major contiguous
    out_ptr,        # pointer to output, shape (Nrows, C) row-major contiguous
    weight_ptr,     # pointer to weight, shape (C,)
    bias_ptr,       # pointer to bias, shape (C,)
    Nrows,          # number of rows (N * D * H * W)
    C,              # number of channels
    eps,            # epsilon for numerical stability (float)
    scale,          # final scaling factor (float)
    ROW_STRIDE,     # number of elements to jump to get to next row start (constexpr or int)
    BLOCK: tl.constexpr  # number of channels handled per program (constexpr)
):
    row_idx = tl.program_id(0)
    # Bounds check for grid (row index)
    if row_idx >= Nrows:
        return

    col_idx = tl.arange(0, BLOCK)
    offs = row_idx * ROW_STRIDE + col_idx  # offsets into flattened (Nrows, C) row-major buffer
    mask = col_idx < C

    # Load a block of channels for this row
    x_vals = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # Compute mean and variance across channels for this row
    sum_x = tl.sum(x_vals)
    sum_x2 = tl.sum(x_vals * x_vals)
    inv_C = 1.0 / C
    mean = sum_x * inv_C
    var = sum_x2 * inv_C - mean * mean
    rsigma = 1.0 / tl.sqrt(var + eps)

    # Normalize
    x_norm = (x_vals - mean) * rsigma

    # Load affine params (weight and bias) for this channel block
    weight = tl.load(weight_ptr + col_idx, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + col_idx, mask=mask, other=0.0)

    # Apply LayerNorm affine
    x_aff = x_norm * weight + bias

    # GELU using erf approximation: 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865476
    y = 0.5 * x_aff * (1.0 + tl.erf(x_aff * inv_sqrt2))

    # Apply final scaling
    y = y * scale

    # Store the result
    tl.store(out_ptr + offs, y, mask=mask)


def triton_fused_layernorm_gelu_scale(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float, scale: float):
    """
    x: Tensor of shape (N, C, D, H, W) (channels-first)
    weight: Tensor of shape (W,)  # normalized over last dimension
    bias: Tensor of shape (W,)
    Returns tensor of same shape with fused LayerNorm (across the last dimension W) + GELU + scaling applied.
    """
    assert x.is_cuda, "Input must be on CUDA"
    assert weight.is_cuda and bias.is_cuda, "LayerNorm params must be on CUDA"

    # Ensure contiguous layout
    x_contig = x.contiguous()
    shape = x_contig.shape  # (N, C, D, H, W)
    # We need to normalize across the last dimension (W) for each (n, c, d, h).
    Nrows = shape[0] * shape[1] * shape[2] * shape[3]
    W_dim = shape[4]

    # Flatten to (Nrows, W)
    x_flat = x_contig.view(Nrows, W_dim)
    out_flat = torch.empty_like(x_flat)

    # Make sure weight/bias are 1D contiguous (they should match W_dim)
    weight1 = weight.contiguous()
    bias1 = bias.contiguous()

    # Grid: one program per row (one row = one (n,c,d,h) position over W)
    grid = (Nrows,)

    # BLOCK is the length of the normalized dimension (W)
    BLOCK = W_dim

    # ROW_STRIDE for row-major contiguous (number of elements per row)
    ROW_STRIDE = W_dim

    # Launch kernel: pass W_dim as the 'C' parameter inside the kernel so it computes
    # mean/variance across the W dimension.
    _fused_ln_gelu_scale_kernel[grid](x_flat, out_flat, weight1, bias1, Nrows, W_dim, eps, scale, ROW_STRIDE, BLOCK=BLOCK)

    # Reshape back to original shape (N, C, D, H, W)
    out = out_flat.view(shape)
    return out


class ModelNew(nn.Module):
    """
    Optimized Model: uses PyTorch ConvTranspose3d for the heavy convolution,
    and a fused Triton kernel for LayerNorm (across channels) + GELU + scaling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, eps=1e-5, scaling_factor=1.0):
        super(ModelNew, self).__init__()
        # Keep the PyTorch ConvTranspose3d for correctness and optimized convolution implementation.
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        # Keep a LayerNorm module to store affine parameters (weight & bias) and eps.
        # We interpret LayerNorm as normalizing across the channel dimension.
        self.layer_norm = nn.LayerNorm(out_channels, eps=eps)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        """
        x: (batch_size, in_channels, D, H, W)
        returns: (batch_size, out_channels, D', H', W')
        """
        x = self.conv_transpose(x)
        # Use the fused Triton kernel to perform LayerNorm (across channels) + GELU + scaling
        # Determine normalization dimension (we normalize across the last dimension W)
        norm_dim = x.shape[-1]
        weight = self.layer_norm.weight if self.layer_norm.elementwise_affine else torch.ones(norm_dim, device=x.device, dtype=x.dtype)
        bias = self.layer_norm.bias if self.layer_norm.elementwise_affine else torch.zeros(norm_dim, device=x.device, dtype=x.dtype)
        out = triton_fused_layernorm_gelu_scale(x, weight, bias, float(self.layer_norm.eps), float(self.scaling_factor))
        return out


# Utility functions matching original module layout
batch_size = 32
in_channels = 32
out_channels = 64
D, H, W = 16, 32, 32
kernel_size = 4
stride = 2
padding = 1
bias = True
eps = 1e-5
scaling_factor = 1.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias, eps, scaling_factor]