import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configurations for the Triton kernel
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 64},  num_warps=2, num_stages=2),
    triton.Config({"BLOCK": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 256}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK": 512}, num_warps=8, num_stages=3),
]

@triton.autotune(
    configs=AUTOTUNE_CONFIGS,
    key=['N', 'C', 'H', 'W', 'out_h', 'out_w', 'kernel', 'stride']
)
@triton.jit
def pool_scale_clamp_kernel(
    x_ptr,           # pointer to input tensor (N, C, H, W)
    scale_ptr,       # pointer to per-channel scale (C,)
    out_ptr,         # pointer to output tensor (N, C, out_h, out_w)
    N, C, H, W,      # input dimensions
    out_h, out_w,    # output spatial dims
    kernel: tl.constexpr, stride: tl.constexpr,  # pooling kernel and stride as constexpr
    clamp_min, clamp_max,  # clamp bounds (floats)
    total_out,       # total number of output elements = N*C*out_h*out_w
    BLOCK: tl.constexpr
):
    # 2D grid:
    #   program_id(0) -> one (n,c) pair
    #   program_id(1) -> a block of spatial outputs (out_h * out_w) processed by this program
    pid_nc = tl.program_id(0)           # index over N*C
    pid_sp = tl.program_id(1)           # index over spatial blocks for this (n,c)

    n = pid_nc // C
    c = pid_nc % C

    # spatial block handled by this program
    block_start = pid_sp * BLOCK
    offs = block_start + tl.arange(0, BLOCK)           # spatial linear indices in [0, out_h*out_w)
    mask_sp = offs < (out_h * out_w)

    # compute (h, w) from spatial linear offset
    w = offs % out_w
    h = offs // out_w

    # compute input window origin for each spatial lane
    in_y0 = h * stride
    in_x0 = w * stride

    # strides for flattening input
    n_stride = C * H * W
    c_stride = H * W
    row_stride = W

    # base index for each lane (points to the top-left of the pooling window for that lane)
    base_idx = n * n_stride + c * c_stride + in_y0 * row_stride + in_x0

    neg_inf = -1e9
    acc = tl.zeros((BLOCK,), dtype=tl.float32) + neg_inf

    # Load scale for this channel once per program and reuse
    scale_val = tl.load(scale_ptr + c)

    # Iterate over the kernel window; kernel is a constexpr so this loop can be unrolled by Triton
    for dy in range(0, kernel):
        row_offset = dy * row_stride
        for dx in range(0, kernel):
            idxs = base_idx + row_offset + dx
            # For valid spatial lanes mask_sp guarantees offsets inside out_h*out_w,
            # and with standard MaxPool (no padding) the corresponding input indices stay in-bounds.
            vals = tl.load(x_ptr + idxs, mask=mask_sp, other=neg_inf)
            vals = vals * scale_val
            cmp = vals > acc
            acc = tl.where(cmp, vals, acc)

    # Apply clamp
    acc = tl.where(acc < clamp_min, clamp_min, acc)
    acc = tl.where(acc > clamp_max, clamp_max, acc)

    # compute flattened output indices corresponding to (n,c,offs)
    out_plane = out_h * out_w
    out_base = n * (C * out_plane) + c * out_plane
    out_idxs = out_base + offs

    tl.store(out_ptr + out_idxs, acc, mask=mask_sp)


class ModelNew(nn.Module):
    """
    Optimized model:
      - Uses built-in PyTorch for Conv2d and GroupNorm (well-optimized)
      - Fuses: scale (per-channel multiply) + MaxPool2d + clamp into a single Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        # keep conv and groupnorm in PyTorch
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        # scale is a learnable parameter
        self.scale = nn.Parameter(torch.ones(scale_shape, dtype=torch.float32))
        # store pooling/clamp params
        self.pool_kernel = maxpool_kernel_size
        self.pool_stride = maxpool_kernel_size  # match PyTorch default when stride is None
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

    def forward(self, x):
        # x: (N, in_channels, H, W)
        x = self.conv(x)
        x = self.group_norm(x)

        # Prepare for Triton kernel:
        # Ensure contiguous tensors on CUDA
        if not x.is_cuda:
            raise RuntimeError("This fused kernel requires CUDA tensors.")
        x = x.contiguous()
        # scale is (C,1,1) -> flatten to (C,)
        scale_flat = self.scale.view(-1).contiguous()

        N, C, H, W = x.shape
        kernel = int(self.pool_kernel)
        stride = int(self.pool_stride)

        # compute output spatial dimensions (consistent with PyTorch MaxPool2d default with no padding)
        out_h = (H - kernel) // stride + 1
        out_w = (W - kernel) // stride + 1

        # allocate output tensor
        out = torch.empty((N, C, out_h, out_w), device=x.device, dtype=x.dtype, requires_grad=x.requires_grad)

        total_out = N * C * out_h * out_w

        # Launch Triton kernel
        # 2D grid: first dimension enumerates (n,c) pairs, second dimension enumerates spatial blocks
        grid = lambda meta: (N * C, (out_h * out_w + meta['BLOCK'] - 1) // meta['BLOCK'])

        pool_scale_clamp_kernel[grid](
            x,                          # x_ptr
            scale_flat,                 # scale_ptr
            out,                        # out_ptr
            N, C, H, W,                 # input dims
            out_h, out_w,               # output spatial dims
            kernel, stride,             # pooling kernel & stride (constexpr in kernel)
            self.clamp_min, self.clamp_max,  # clamp bounds
            total_out                   # total number of output elements
        )

        return out