import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configs tuned for A6000 (autotune over channel-block and spatial-block)
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_C": 64,  "BLOCK_SPATIAL": 64},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_C": 128, "BLOCK_SPATIAL": 64},  num_warps=4, num_stages=3),
    triton.Config({"BLOCK_C": 128, "BLOCK_SPATIAL": 128}, num_warps=8, num_stages=3),
]

@triton.autotune(
    configs=AUTOTUNE_CONFIGS,
    key=['C', 'D']
)
@triton.jit
def fused_mean_softmax_tanh_kernel(
    x_ptr,          # pointer to input (B, C, D, H, W) flattened
    bias_ptr,       # pointer to bias (C,)
    out_ptr,        # pointer to output (B, C, H, W) flattened
    B, C, D, H, W,  # dims
    scaling,
    BLOCK_C: tl.constexpr,
    BLOCK_SPATIAL: tl.constexpr,
):
    """
    Triton kernel that:
      - Processes a block of channels (BLOCK_C) and a block of spatial positions (BLOCK_SPATIAL)
      - Computes mean over depth D (vectorized in small unrolled steps)
      - Adds per-channel bias
      - Performs numerically-stable softmax across channels (per spatial position)
      - Applies tanh (via exp trick) and scaling
    Writes output in shape (B, C, H, W).
    Grid layout:
      program_id(0) -> channel block
      program_id(1) -> combined (batch, spatial-block)
    """
    pid_c = tl.program_id(0)   # channel block index
    pid_sp = tl.program_id(1)  # combined batch * spatial-block index

    HW = H * W
    num_sp_blocks = (HW + BLOCK_SPATIAL - 1) // BLOCK_SPATIAL
    b = pid_sp // num_sp_blocks
    sp_block = pid_sp - b * num_sp_blocks

    # spatial offsets handled in a vector inside each program
    sp_start = sp_block * BLOCK_SPATIAL
    offs_sp = sp_start + tl.arange(0, BLOCK_SPATIAL)              # (BLOCK_SPATIAL,)
    mask_sp = offs_sp < HW

    h = offs_sp // W
    w = offs_sp - h * W
    # base per-spatial offset (within a batch)
    base_sp = h * W + w  # equals h*stride_H_in + w*stride_W_in

    # strides for input (B, C, D, H, W)
    stride_B_in = C * D * H * W
    stride_C_in = D * H * W
    stride_D_in = H * W
    # strides for output (B, C, H, W)
    stride_B_out = C * HW
    stride_C_out = HW

    # channel offsets within block
    offs_c = tl.arange(0, BLOCK_C)  # (BLOCK_C,)
    c_start = pid_c * BLOCK_C
    abs_c = c_start + offs_c       # (BLOCK_C,)
    mask_c = abs_c < C             # (BLOCK_C,)

    # prepare 2D masks and offsets with broadcasting: (BLOCK_C, BLOCK_SPATIAL)
    mask2d = mask_c[:, None] & mask_sp[None, :]

    # base address components
    base_B = b * stride_B_in
    base_in = base_B + base_sp[None, :]                # (1, BLOCK_SPATIAL) broadcast later
    offs_in = base_in + (abs_c[:, None] * stride_C_in) # (BLOCK_C, BLOCK_SPATIAL)

    # accumulator: sum over depths for each (channel, spatial)
    vals = tl.zeros((BLOCK_C, BLOCK_SPATIAL), dtype=tl.float32)

    # Vectorized depth reduction: unroll 4 depths per iteration to reduce loop overhead
    d = 0
    while d < D:
        # unroll up to 4 depths per iteration
        # depth 0
        idx = d
        if idx < D:
            ptrs = offs_in + idx * stride_D_in    # (BLOCK_C, BLOCK_SPATIAL)
            cur = tl.load(x_ptr + ptrs, mask=mask2d, other=0.0)
            vals += cur
        # depth 1
        idx = d + 1
        if idx < D:
            ptrs = offs_in + idx * stride_D_in
            cur = tl.load(x_ptr + ptrs, mask=mask2d, other=0.0)
            vals += cur
        # depth 2
        idx = d + 2
        if idx < D:
            ptrs = offs_in + idx * stride_D_in
            cur = tl.load(x_ptr + ptrs, mask=mask2d, other=0.0)
            vals += cur
        # depth 3
        idx = d + 3
        if idx < D:
            ptrs = offs_in + idx * stride_D_in
            cur = tl.load(x_ptr + ptrs, mask=mask2d, other=0.0)
            vals += cur
        d += 4

    # compute mean over depth
    inv_D = 1.0 / D
    vals = vals * inv_D  # (BLOCK_C, BLOCK_SPATIAL)

    # load bias per channel and broadcast across spatial
    bias_vals = tl.load(bias_ptr + abs_c, mask=mask_c, other=0.0)  # (BLOCK_C,)
    vals = vals + bias_vals[:, None]

    # For numerical stability, set masked positions very negative before max
    neg_inf = tl.full((BLOCK_C, BLOCK_SPATIAL), -1e20, dtype=tl.float32)
    vals_for_max = tl.where(mask2d, vals, neg_inf)

    # softmax across channel axis (axis=0)
    max_v = tl.max(vals_for_max, axis=0)                # (BLOCK_SPATIAL,)
    exps = tl.exp(vals - max_v[None, :])
    exps = tl.where(mask2d, exps, 0.0)
    sum_v = tl.sum(exps, axis=0) + 1e-6                # (BLOCK_SPATIAL,)
    soft = exps / sum_v[None, :]

    # tanh via exp trick and scaling
    e_pos = tl.exp(soft)
    e_neg = tl.exp(-soft)
    tanh_vals = (e_pos - e_neg) / (e_pos + e_neg)
    out_vals = tanh_vals * scaling                      # (BLOCK_C, BLOCK_SPATIAL)

    # store results to output (B, C, H, W)
    base_out = b * stride_B_out + base_sp[None, :]      # (1, BLOCK_SPATIAL)
    offs_out = base_out + abs_c[:, None] * stride_C_out # (BLOCK_C, BLOCK_SPATIAL)
    tl.store(out_ptr + offs_out, out_vals, mask=mask2d)


def fused_mean_softmax_tanh(x: torch.Tensor, bias: torch.Tensor, scaling: float):
    """
    x: (B, C, D, H, W) contiguous cuda tensor (float32)
    bias: (C,) contiguous cuda tensor
    returns: (B, C, H, W) cuda tensor after softmax (over channels), tanh and scaling
    (All fused inside Triton kernel for performance.)
    """
    assert x.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    bias = bias.contiguous()

    B, C, D, H, W = x.shape
    out = torch.empty((B, C, H, W), device=x.device, dtype=x.dtype)

    num_sp = H * W
    # Default block choices; autotune will override via configs
    BLOCK_C_DEF = 128
    BLOCK_SPATIAL_DEF = 64

    num_c_blocks = (C + BLOCK_C_DEF - 1) // BLOCK_C_DEF
    num_sp_blocks = (num_sp + BLOCK_SPATIAL_DEF - 1) // BLOCK_SPATIAL_DEF
    # grid: (channel_blocks, batch * spatial_blocks)
    grid = lambda meta: (
        (C + meta["BLOCK_C"] - 1) // meta["BLOCK_C"],
        B * ((num_sp + meta["BLOCK_SPATIAL"] - 1) // meta["BLOCK_SPATIAL"]),
    )

    # Launch fused kernel (autotune will supply BLOCK_C and BLOCK_SPATIAL)
    fused_mean_softmax_tanh_kernel[grid](x, bias, out, B, C, D, H, W, float(scaling))
    return out


class ModelNew(nn.Module):
    """
    Optimized Model that:
      - Keeps PyTorch ConvTranspose3d for the heavy convolution transpose.
      - Uses a Triton kernel that fuses:
          mean over depth D, bias add, softmax over channels, tanh, and scaling,
        avoiding intermediate allocation for the depth-reduced tensor.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        # keep bias shape (1, C, 1, 1, 1) as original
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        # x: (B, in_channels, D, H, W)
        x = self.conv_transpose(x)                 # (B, C, D, H, W)
        # prepare bias as (C,)
        bias_vec = self.bias.view(self.bias.size(1)).contiguous()
        # fused kernel computes mean over D internally and returns (B, C, H, W)
        out_squeezed = fused_mean_softmax_tanh(x, bias_vec, self.scaling_factor)
        # restore depth dimension to match original expected output shape (B, C, 1, H, W)
        out = out_squeezed.unsqueeze(2)
        return out