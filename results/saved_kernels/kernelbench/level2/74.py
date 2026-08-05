import torch
import torch.nn as nn
import triton
import triton.language as tl

# Triton kernel: fuse LeakyReLU -> (max/min trick + single multiply + single LeakyReLU) -> 3D maxpool (2x2x2)
@triton.jit
def fused_act_mul_pool_kernel(
    x_ptr,          # pointer to input tensor (N, C, D_in, H_in, W_in) flattened
    mult_ptr,       # pointer to multiplier (C,) flattened
    out_ptr,        # pointer to output tensor (N, C, D_out, H_out, W_out) flattened
    N, C, D_in, H_in, W_in, D_out, H_out, W_out,
    n_elements,     # number of output elements
    neg_slope,      # leaky relu negative slope
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements

    # Compute multi-dimensional indices from flattened output offset
    # offsets correspond to linear indices in layout: (((n*C + c)*D_out + d)*H_out + h)*W_out + w
    w = offs % W_out
    tmp = offs // W_out
    h = tmp % H_out
    tmp = tmp // H_out
    d = tmp % D_out
    tmp = tmp // D_out
    c = tmp % C
    n = tmp // C

    # Compute input coordinates for the 2x2x2 pooling window
    d0 = d * 2
    h0 = h * 2
    w0 = w * 2

    # Precompute factors for flattening input indices (N, C, D_in, H_in, W_in)
    # flat_index = (((n * C + c) * D_in + d_in) * H_in + h_in) * W_in + w_in
    nc = n * C + c
    base_ncd = nc * D_in  # will be added with d_in

    # compute base flattened index for (n,c,d0,h0,w0)
    base_idx = ((base_ncd + d0) * H_in + h0) * W_in + w0

    # small strides to reach neighbors
    W_stride = 1
    H_stride = W_in        # adding 1 to h increases flat index by W_in
    D_stride = H_in * W_in # adding 1 to d increases flat index by H_in * W_in

    # offsets for the 8 neighbors relative to base_idx:
    # (0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)
    o0 = 0
    o1 = W_stride
    o2 = H_stride
    o3 = H_stride + W_stride
    o4 = D_stride
    o5 = D_stride + W_stride
    o6 = D_stride + H_stride
    o7 = D_stride + H_stride + W_stride

    neg_inf = -1e20

    v000 = tl.load(x_ptr + base_idx + o0, mask=mask, other=neg_inf)
    v001 = tl.load(x_ptr + base_idx + o1, mask=mask, other=neg_inf)
    v010 = tl.load(x_ptr + base_idx + o2, mask=mask, other=neg_inf)
    v011 = tl.load(x_ptr + base_idx + o3, mask=mask, other=neg_inf)
    v100 = tl.load(x_ptr + base_idx + o4, mask=mask, other=neg_inf)
    v101 = tl.load(x_ptr + base_idx + o5, mask=mask, other=neg_inf)
    v110 = tl.load(x_ptr + base_idx + o6, mask=mask, other=neg_inf)
    v111 = tl.load(x_ptr + base_idx + o7, mask=mask, other=neg_inf)

    # First LeakyReLU (applied to each neighbor)
    v000 = tl.where(v000 > 0.0, v000, v000 * neg_slope)
    v001 = tl.where(v001 > 0.0, v001, v001 * neg_slope)
    v010 = tl.where(v010 > 0.0, v010, v010 * neg_slope)
    v011 = tl.where(v011 > 0.0, v011, v011 * neg_slope)
    v100 = tl.where(v100 > 0.0, v100, v100 * neg_slope)
    v101 = tl.where(v101 > 0.0, v101, v101 * neg_slope)
    v110 = tl.where(v110 > 0.0, v110, v110 * neg_slope)
    v111 = tl.where(v111 > 0.0, v111, v111 * neg_slope)

    # Compute max and min across the 8 neighbors (needed for sign-aware scaling)
    m01 = tl.maximum(v000, v001)
    m23 = tl.maximum(v010, v011)
    m45 = tl.maximum(v100, v101)
    m67 = tl.maximum(v110, v111)
    max_y = tl.maximum(tl.maximum(m01, m23), tl.maximum(m45, m67))

    n01 = tl.minimum(v000, v001)
    n23 = tl.minimum(v010, v011)
    n45 = tl.minimum(v100, v101)
    n67 = tl.minimum(v110, v111)
    min_y = tl.minimum(tl.minimum(n01, n23), tl.minimum(n45, n67))

    # Load multiplier per channel once (broadcasted across lanes)
    m = tl.load(mult_ptr + c, mask=mask, other=1.0)

    # Select proper scaled value depending on sign of m:
    # if m >= 0: m * max_y else m * min_y
    m_nonneg = m >= 0.0
    scaled = tl.where(m_nonneg, m * max_y, m * min_y)

    # Single LeakyReLU applied after scaling
    out_vals = tl.where(scaled > 0.0, scaled, scaled * neg_slope)

    # Store results
    tl.store(out_ptr + offs, out_vals, mask=mask)


def fused_act_mul_pool(x: torch.Tensor, multiplier: torch.Tensor, neg_slope: float = 0.2):
    """
    Wrapper that prepares tensors and launches the Triton kernel.
    x: Tensor of shape (N, C, D_in, H_in, W_in)
    multiplier: Tensor of shape (C, 1, 1, 1) or (C,)
    Returns: Tensor of shape (N, C, D_out, H_out, W_out), where D_out = D_in // 2, etc.
    """
    assert x.is_cuda and multiplier.is_cuda, "Inputs must be on CUDA"
    assert x.dtype == torch.float32 and multiplier.dtype == torch.float32

    x = x.contiguous()
    # Ensure multiplier is 1D of length C
    if multiplier.dim() != 1:
        multiplier1d = multiplier.view(multiplier.shape[0]).contiguous()
    else:
        multiplier1d = multiplier.contiguous()

    N, C, D_in, H_in, W_in = x.shape
    assert D_in % 2 == 0 and H_in % 2 == 0 and W_in % 2 == 0, "Input spatial dims must be divisible by 2 for 2x2x2 pooling"

    D_out = D_in // 2
    H_out = H_in // 2
    W_out = W_in // 2

    out = torch.empty((N, C, D_out, H_out, W_out), device=x.device, dtype=x.dtype)
    n_elements = out.numel()

    # Flatten pointers
    x_ptr = x.reshape(-1)
    out_ptr = out.reshape(-1)
    mult_ptr = multiplier1d.reshape(-1)

    # Configure block size (tuned)
    BLOCK = 128
    grid = ( (n_elements + BLOCK - 1) // BLOCK, )

    # Launch kernel
    fused_act_mul_pool_kernel[grid](
        x_ptr, mult_ptr, out_ptr,
        N, C, D_in, H_in, W_in, D_out, H_out, W_out,
        n_elements, neg_slope,
        BLOCK=BLOCK
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized model: performs ConvTranspose3d (PyTorch), then a fused Triton kernel that:
      LeakyReLU -> multiply by per-channel parameter -> LeakyReLU -> 3D maxpool (2x2x2)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                                 stride=stride, padding=padding, output_padding=output_padding)
        # multiplier stored in same shape as original for compatibility, but kernel uses a flattened view
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.neg_slope = 0.2

    def forward(self, x):
        # conv_transpose done by PyTorch (keeps using optimized CuDNN)
        x = self.conv_transpose(x)
        # Fuse activations, scaling and pooling in Triton
        return fused_act_mul_pool(x, self.multiplier, neg_slope=self.neg_slope)


# Keep input generation functions for compatibility with original interface
batch_size = 16
in_channels = 16
out_channels = 32
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
multiplier_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda().float()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape]