import torch
import torch.nn as nn
import triton
import triton.language as tl


# Triton kernels to accelerate min-reduction over depth and channel-wise softmax

@triton.jit
def _min_reduce_kernel(
    inp_ptr,            # pointer to input tensor (B, C, D, H, W)
    out_ptr,            # pointer to output tensor (B, C, H, W) - min reduced over D
    B, C, D, H, W,
    stride_b, stride_c, stride_d, stride_h, stride_w,
    n_elements,         # total number of elements in out (B*C*H*W)
    BLOCK: tl.constexpr,    # number of flattened elements (over B*C*H*W) handled per program
    MAX_D: tl.constexpr,    # maximum D loop bound (constexpr)
):
    """
    Each program handles BLOCK flattened output elements. For each output element (b,c,h,w),
    compute min over d in [0, D) on input[b, c, d, h, w] and write to out[b, c, h, w].
    """
    # flattened index handled by this program
    idx = tl.arange(0, BLOCK)
    start = tl.program_id(0) * BLOCK
    off = start + idx  # flattened indices
    mask = off < n_elements

    # Compute multi-dimensional indices from flattened index off:
    # layout: off = b*(C*H*W) + c*(H*W) + h*(W) + w
    CHW = C * H * W
    HW = H * W

    b = off // CHW
    rem = off % CHW
    c = rem // HW
    rem2 = rem % HW
    h = rem2 // W
    w = rem2 % W

    # Compute base pointer offsets per element (points to d=0 location for each element)
    base_offset = b * stride_b + c * stride_c + h * stride_h + w * stride_w

    # Initialize min values with a large positive number
    min_val = tl.full((BLOCK,), 1e30, dtype=tl.float32)

    # Loop over depth up to MAX_D and only consider d < D
    for d in range(MAX_D):
        valid_d = d < D
        # compute pointer for this d
        ptrs = inp_ptr + base_offset + d * stride_d
        # masked load: only valid lanes and valid depth positions load real data; others get large value
        m = mask & valid_d
        vals = tl.load(ptrs, mask=m, other=1e30)
        # elementwise min
        min_val = tl.minimum(min_val, vals)

    # Store results back to out (contiguous flattened layout assumed)
    out_ptr_off = out_ptr + off
    tl.store(out_ptr_off, min_val, mask=mask)


@triton.jit
def _softmax_channel_kernel(
    inp_ptr,            # pointer to input tensor (B, C, H, W) - the min-reduced tensor
    out_ptr,            # pointer to output tensor (B, C, H, W) - softmax over channels
    B, C, H, W,
    stride_b, stride_c, stride_h, stride_w,
    n_locs,             # number of (b,h,w) locations = B*H*W
    BLOCK_POS: tl.constexpr,  # number of (b,h,w) positions handled per program
    MAX_C: tl.constexpr,      # maximum channels bound (constexpr)
):
    """
    Each program handles BLOCK_POS positions (b,h,w). For each position, compute softmax over channels C:
      out[b, c, h, w] = softmax_c(inp[b, c, h, w])
    """
    pos_idx = tl.arange(0, BLOCK_POS)
    start = tl.program_id(0) * BLOCK_POS
    pos = start + pos_idx
    mask_pos = pos < n_locs

    # Compute b,h,w from pos flattened as pos = b*(H*W) + h*W + w
    HW = H * W
    b = pos // HW
    rem = pos % HW
    h = rem // W
    w = rem % W

    # Compute base pointer offsets per position (for channel offset 0)
    base_offset = b * stride_b + h * stride_h + w * stride_w

    # First pass: compute max over channels for numerical stability
    max_val = tl.full((BLOCK_POS,), -1e30, dtype=tl.float32)
    for c in range(MAX_C):
        valid_c = c < C
        ptrs = inp_ptr + base_offset + c * stride_c
        m = mask_pos & valid_c
        vals = tl.load(ptrs, mask=m, other=-1e30)
        max_val = tl.maximum(max_val, vals)

    # Second pass: compute sum of exp(vals - max_val)
    sum_exp = tl.zeros((BLOCK_POS,), dtype=tl.float32)
    for c in range(MAX_C):
        valid_c = c < C
        ptrs = inp_ptr + base_offset + c * stride_c
        m = mask_pos & valid_c
        vals = tl.load(ptrs, mask=m, other=0.0)
        # subtract max and exponentiate
        exp_vals = tl.exp(vals - max_val)
        # for invalid lanes (mask false) exp_vals will be something but m masks accumulation
        sum_exp = sum_exp + tl.where(m, exp_vals, 0.0)

    # Third pass: write normalized softmax outputs back
    for c in range(MAX_C):
        valid_c = c < C
        ptrs_in = inp_ptr + base_offset + c * stride_c
        ptrs_out = out_ptr + base_offset + c * stride_c
        m = mask_pos & valid_c
        vals = tl.load(ptrs_in, mask=m, other=0.0)
        out_vals = tl.exp(vals - max_val) / sum_exp
        # When sum_exp might be zero (shouldn't be), safe division is assumed as sum_exp>0 for valid lanes
        tl.store(ptrs_out, out_vals, mask=m)


class ModelNew(nn.Module):
    """
    Optimized version of the original Model using Triton kernels to fuse:
      - min reduction over the depth dimension (D)
      - channel-wise softmax over the channel dimension (C)
    The 3D convolution is kept as PyTorch's efficient implementation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.dim = dim  # expected to be 2 (depth dimension) in the provided setup

        # Tunable constants for the Triton kernels
        # These should be >= the maximum runtime values for D and C in the target workload.
        # We set conservative static upper-bounds based on the problem description.
        self._triton_max_d = 32   # D' after conv is expected <= 22 -> 32 is safe
        self._triton_max_c = 32   # out_channels is 24 -> 32 is safe

        # Kernel block sizes (tunable for performance)
        self._min_block = 1024    # number of flattened elements per Triton program for min kernel
        self._softmax_block_pos = 64  # number of positions (b,h,w) per Triton program for softmax

    def forward(self, x):
        """
        Forward pass:
          1. Apply 3D convolution (PyTorch)
          2. Compute min over depth dimension (self.dim) using Triton kernel -> temp tensor shape (B, C, H, W)
          3. Compute softmax over channels using Triton kernel -> output shape (B, C, H, W)
        """
        # 1) convolution
        x = self.conv(x)  # shape: (B, C, D, H, W)

        # Ensure input is on CUDA and contiguous
        if not x.is_cuda:
            # Fallback to CPU PyTorch ops if no CUDA is available
            # Perform min over depth then softmax using PyTorch
            reduced = torch.min(x, dim=self.dim)[0]
            return torch.softmax(reduced, dim=1)

        x = x.contiguous()

        B, C, D, H, W = x.shape

        # 2) Min-reduction over depth -> temp tensor (B, C, H, W)
        temp = torch.empty((B, C, H, W), device=x.device, dtype=x.dtype)

        # Compute strides in element counts
        # For tensor of shape (B, C, D, H, W) with contiguous memory, strides correspond to element strides.
        s_in = x.stride()
        stride_b_in = s_in[0]
        stride_c_in = s_in[1]
        stride_d_in = s_in[2]
        stride_h_in = s_in[3]
        stride_w_in = s_in[4]

        # For temp tensor (B, C, H, W), compute its contiguous strides
        temp = temp.contiguous()
        s_temp = temp.stride()
        stride_b_temp = s_temp[0]
        stride_c_temp = s_temp[1]
        stride_h_temp = s_temp[2]
        stride_w_temp = s_temp[3]

        n_out_elements = B * C * H * W

        # Launch min-reduction Triton kernel
        grid_min = ( (n_out_elements + self._min_block - 1) // self._min_block, )
        _min_reduce_kernel[grid_min](
            x,                      # inp_ptr
            temp,                   # out_ptr
            B, C, D, H, W,
            stride_b_in, stride_c_in, stride_d_in, stride_h_in, stride_w_in,
            n_out_elements,
            BLOCK=self._min_block,
            MAX_D=self._triton_max_d
        )

        # 3) Softmax across channels: operate on temp (B, C, H, W)
        # Prepare output tensor
        out = torch.empty_like(temp)

        # Strides for temp/out (they are contiguous)
        s_temp = temp.stride()
        stride_b = s_temp[0]
        stride_c = s_temp[1]
        stride_h = s_temp[2]
        stride_w = s_temp[3]

        n_positions = B * H * W
        grid_softmax = ( (n_positions + self._softmax_block_pos - 1) // self._softmax_block_pos, )

        _softmax_channel_kernel[grid_softmax](
            temp,
            out,
            B, C, H, W,
            stride_b, stride_c, stride_h, stride_w,
            n_positions,
            BLOCK_POS=self._softmax_block_pos,
            MAX_C=self._triton_max_c
        )

        return out


# Helper factory functions for compatibility with the testing harness
def get_inputs():
    batch_size = 128
    in_channels = 3
    D, H, W = 24, 32, 32
    # single input tensor
    return [torch.rand(batch_size, in_channels, D, H, W).cuda()]


def get_init_inputs():
    in_channels = 3
    out_channels = 24
    kernel_size = 3
    dim = 2
    return [in_channels, out_channels, kernel_size, dim]