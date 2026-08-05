import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configs for the fused kernel - BLOCKs chosen to be multiples of 32 (warp-aligned)
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 64},   num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 128},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 256},  num_warps=8, num_stages=2),
    triton.Config({"BLOCK": 512},  num_warps=8, num_stages=3),
    triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=3),
]

# The fused kernel:
# For each (batch, channel) pair (one program), this kernel:
#  - performs 2x2 max-pooling with stride 2 over HxW
#  - applies hardtanh (clamp)
#  - computes the mean over pooled spatial locations
# It writes one scalar mean per (batch,channel) into out_ptr (flattened N*C array).
@triton.autotune(
    configs=AUTOTUNE_CONFIGS,
    key=["N", "C", "pooled_total"],
)
@triton.jit
def fused_pool_clamp_mean_kernel(
    inp_ptr,          # pointer to input tensor (N, C, H, W), flattened
    out_ptr,          # pointer to output tensor (N*C,) flattened
    N, C, H, W,       # input dimensions
    pooled_H, pooled_W, pooled_total,  # pooling dimensions and total pooled elements
    clamp_min, clamp_max,              # hardtanh bounds (fp32)
    BLOCK: tl.constexpr,
):
    # 2D grid: program_id(0)=n (batch), program_id(1)=c (channel)
    n = tl.program_id(0)
    c = tl.program_id(1)
    # base linear offset to (n, c, 0, 0)
    base = ((n * C + c) * H) * W

    # accumulator for the sum of all pooled values (keep as python float -> treated as fp32 accumulation)
    acc = 0.0

    # Iterate pooled rows outer and tile pooled columns inner for coalesced loads:
    for ph in range(pooled_H):
        base_row = base + (ph * 2) * W  # top-left row offset for this pooled row
        # inner tiled loop over pooled_W with tile size BLOCK (constexpr)
        for w_off in range(0, pooled_W, BLOCK):
            offs = tl.arange(0, BLOCK)              # [0, 1, ..., BLOCK-1]
            w = w_off + offs                        # pooled column indices for this tile
            mask = w < pooled_W                     # valid columns in this tile

            # compute linear offsets for the top-left of each 2x2 block (contiguous across columns)
            block_base = base_row + w               # addresses for top-left elements

            # load four values of the 2x2 block for each pooled element (coalesced across columns)
            a = tl.load(inp_ptr + block_base, mask=mask, other=clamp_min)
            b = tl.load(inp_ptr + block_base + 1, mask=mask, other=clamp_min)
            c2 = tl.load(inp_ptr + block_base + W, mask=mask, other=clamp_min)
            d = tl.load(inp_ptr + block_base + W + 1, mask=mask, other=clamp_min)

            # elementwise 2x2 max
            m1 = tl.maximum(a, b)
            m2 = tl.maximum(c2, d)
            m = tl.maximum(m1, m2)

            # hardtanh clamp
            m_clamped = tl.minimum(tl.maximum(m, clamp_min), clamp_max)

            # zero out invalid lanes and reduce
            valid = tl.where(mask, m_clamped, 0.0)
            s = tl.sum(valid)
            acc += s

    # compute mean (scalar)
    mean = acc / pooled_total

    # store the result into out_ptr at position n*C + c
    out_idx = n * C + c
    tl.store(out_ptr + out_idx, mean)


def triton_fused_pool_clamp_mean(inp: torch.Tensor, clamp_min: float, clamp_max: float):
    """
    Wrapper that prepares tensors and launches the Triton fused kernel.
    Input: inp (N, C, H, W)
    Output: (N, C, 1, 1) with mean over pooled spatial dims (after pooling+clamp).
    """
    assert inp.is_cuda, "Input must be a CUDA tensor"
    inp = inp.contiguous()
    N, C, H, W = inp.shape
    # Only support even H and W (since pooling 2x2 stride 2)
    assert H % 2 == 0 and W % 2 == 0, "H and W must be divisible by 2 for 2x2 pooling"

    pooled_H = H // 2
    pooled_W = W // 2
    pooled_total = pooled_H * pooled_W

    # prepare output flattened (N*C,)
    out = torch.empty((N * C,), dtype=inp.dtype, device=inp.device)

    # grid: 2D grid (N, C) -> program_id(0)=n, program_id(1)=c
    grid = lambda meta: (N, C)

    # launch kernel
    fused_pool_clamp_mean_kernel[grid](
        inp,                 # inp_ptr
        out,                 # out_ptr
        N, C, H, W,
        pooled_H, pooled_W, pooled_total,
        float(clamp_min), float(clamp_max),
    )

    # reshape to (N, C, 1, 1)
    out = out.view(N, C, 1, 1)
    return out


class ModelNew(nn.Module):
    """
    Optimized Model that uses Triton to fuse MaxPool2d(kernel=2,stride=2) + Hardtanh + mean over H,W.
    The ConvTranspose2d is kept as the PyTorch operator for correctness and leverage of cuDNN.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        # Keep the transposed convolution as PyTorch operator
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        # store pooling and clamp params (pooling assumed 2x2 stride 2 for the fused kernel)
        assert maxpool_kernel_size == 2 and maxpool_stride == 2, "This fused kernel targets 2x2 maxpool with stride 2"
        self.hardtanh_min = float(hardtanh_min)
        self.hardtanh_max = float(hardtanh_max)

    def forward(self, x):
        # conv transpose (use PyTorch's implementation)
        x = self.conv_transpose(x)  # (N, C, H, W)

        # fuse MaxPool2d(kernel=2,stride=2) + Hardtanh + mean over (H,W) using Triton kernel
        x = triton_fused_pool_clamp_mean(x, self.hardtanh_min, self.hardtanh_max)  # (N, C, 1, 1)

        # final tanh activation (on small tensor, use PyTorch)
        x = torch.tanh(x)
        return x