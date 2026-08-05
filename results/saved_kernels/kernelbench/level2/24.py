import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Triton kernel: reduce (min) over the D dimension of a 5D tensor (B, C, D, H, W)
# Produces output of shape (B, C, H, W)
@triton.jit
def _min_reduce_d_kernel(
    x_ptr,             # pointer to input tensor (fp32)
    out_ptr,           # pointer to output tensor (fp32)
    B, C, D, H, W,     # sizes
    stride_b, stride_c, stride_d, stride_h, stride_w,  # strides (in elements)
    n_elements,        # total number of elements in output = B*C*H*W
    BLOCK: tl.constexpr,  # number of elements handled per program
):
    pid = tl.program_id(0)
    # compute offsets this program will handle
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    # compute (b, c, h, w) for each offset
    tmp = C * H * W
    b = offsets // tmp
    rem = offsets - b * tmp
    ch_size = H * W
    c = rem // ch_size
    rem2 = rem - c * ch_size
    h = rem2 // W
    w = rem2 - h * W

    # broadcast vectors for pointer computation
    # compute base pointers per output lane (avoid recomputing b/c/h/w arithmetic each depth)
    base_ptrs = x_ptr + b * stride_b + c * stride_c + h * stride_h + w * stride_w
    # initialize min_vals with values at d=0 (or +inf for out-of-bounds lanes)
    ptrs = base_ptrs + 0 * stride_d
    min_vals = tl.load(ptrs, mask=mask, other=1e9)

    # iterate over remaining depth slices, using base_ptrs to avoid repeated arithmetic
    # Note: D is a kernel argument; using a Python loop over range(D) is allowed here.
    for d in range(1, D):
        ptrs = base_ptrs + d * stride_d
        vals = tl.load(ptrs, mask=mask, other=1e9)
        min_vals = tl.minimum(min_vals, vals)

    # store results into output (flattened by offsets)
    tl.store(out_ptr + offsets, min_vals, mask=mask)


def triton_min_reduce_d(x: torch.Tensor):
    """
    Wrapper for the Triton min-reduction kernel over the D dimension.
    Expects x of shape (B, C, D, H, W).
    Returns tensor of shape (B, C, H, W).
    """
    assert x.is_cuda, "Input must be on CUDA"
    assert x.dtype == torch.float32, "Only fp32 is supported by this kernel"

    # Ensure contiguous memory layout for simple stride passing
    x = x.contiguous()

    B, C, D, H, W = x.shape
    # output shape
    out = torch.empty((B, C, H, W), device=x.device, dtype=x.dtype)

    # strides in elements
    sB, sC, sD, sH, sW = x.stride()

    n_elements = B * C * H * W

    # Choose a block size (constexpr). Use a smaller BLOCK to improve occupancy on Ampere.
    BLOCK = 256
    grid = ( (n_elements + BLOCK - 1) // BLOCK, )

    # Launch the kernel
    _min_reduce_d_kernel[grid](
        x, out,
        B, C, D, H, W,
        sB, sC, sD, sH, sW,
        n_elements,
        BLOCK=BLOCK
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model: uses the PyTorch Conv3d for the convolution, then uses a Triton kernel
    to compute the minimum over the specified dimension when that dimension is the depth (D).
    Finally, applies softmax over the channel dimension.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.dim = dim  # expected to be 2 for depth (D). If not 2, falls back to torch.min.

    def forward(self, x):
        """
        x: (batch_size, in_channels, D, H, W)
        returns: (batch_size, out_channels, H, W)
        """
        # Compute the full conv3d output once, then reduce along depth.
        out_full = self.conv(x)  # (B, C, D_out, H_out, W_out)
        if self.dim == 2:
            # Use Triton min-reduction over depth (D_out)
            x_min = triton_min_reduce_d(out_full)
        else:
            # Fallback for other dims
            x_min = torch.min(out_full, dim=self.dim)[0]
        out = torch.softmax(x_min, dim=1)
        return out


# Keep the original helper functions (inputs)
batch_size = 128
in_channels = 3
out_channels = 24  # Increased output channels
D, H, W = 24, 32, 32  # Increased depth
kernel_size = 3
dim = 2  # Dimension along which to apply minimum operation (e.g., depth)

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W).cuda().float()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, dim]