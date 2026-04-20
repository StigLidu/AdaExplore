import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configs to pick the best BLOCK and number of warps for Ampere (A6000)
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 64},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 256}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK": 512}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=3),
]

@triton.autotune(
    configs=AUTOTUNE_CONFIGS,
    key=["N", "C", "H", "W", "pooled_H", "pooled_W"],
)
@triton.jit
def _fused_sub_hswish_pool_mish_kernel(
    inp_ptr,            # input pointer (conv output) [N, C, H, W] flattened
    out_ptr,            # output pointer [N, C, pooled_H, pooled_W] flattened
    N, C, H, W,         # input dims
    pooled_H, pooled_W, # pooled output spatial dims
    subtract_val,       # scalar to subtract (float)
    numel,              # total number of output elements (N*C*pooled_H*pooled_W)
    BLOCK: tl.constexpr  # number of pooled positions processed per program
):
    pid = tl.program_id(0)

    # total pooled positions per (n,c) plane
    P = pooled_H * pooled_W
    # tiles per plane
    tiles_per_plane = (P + BLOCK - 1) // BLOCK

    # which (n, c) plane and which tile within that plane
    plane = pid // tiles_per_plane
    tile = pid % tiles_per_plane

    # range of pooled positions this program handles
    offs = tl.arange(0, BLOCK)
    idx_in_plane = tile * BLOCK + offs
    mask_pos = idx_in_plane < P

    # compute pooled coords
    ph = idx_in_plane // pooled_W
    pw = idx_in_plane % pooled_W

    # derive n and c from plane id
    n = plane // C
    c = plane % C

    # top-left corner in input for each pooled cell (stride 2 pooling)
    i0 = ph * 2
    j0 = pw * 2

    # four positions in 2x2 pooling window
    i00 = i0
    j00 = j0
    i01 = i0
    j01 = j0 + 1
    i10 = i0 + 1
    j10 = j0
    i11 = i0 + 1
    j11 = j0 + 1

    # bounds checks for each position
    valid00 = (i00 < H) & (j00 < W) & mask_pos
    valid01 = (i01 < H) & (j01 < W) & mask_pos
    valid10 = (i10 < H) & (j10 < W) & mask_pos
    valid11 = (i11 < H) & (j11 < W) & mask_pos

    # compute linear indices into input flattened: ((n*C + c) * H + i) * W + j
    base_plane = ((n * C) + c) * H
    idx00 = (base_plane + i00) * W + j00
    idx01 = (base_plane + i01) * W + j01
    idx10 = (base_plane + i10) * W + j10
    idx11 = (base_plane + i11) * W + j11

    # load values (use a large negative for out-of-bounds so max pooling ignores them)
    neg_inf = -1e30
    v00 = tl.load(inp_ptr + idx00, mask=valid00, other=neg_inf)
    v01 = tl.load(inp_ptr + idx01, mask=valid01, other=neg_inf)
    v10 = tl.load(inp_ptr + idx10, mask=valid10, other=neg_inf)
    v11 = tl.load(inp_ptr + idx11, mask=valid11, other=neg_inf)

    # subtract scalar
    sv00 = v00 - subtract_val
    sv01 = v01 - subtract_val
    sv10 = v10 - subtract_val
    sv11 = v11 - subtract_val

    # HardSwish: x * relu6(x + 3) / 6
    a00 = sv00 + 3.0
    a01 = sv01 + 3.0
    a10 = sv10 + 3.0
    a11 = sv11 + 3.0

    # clamp to [0, 6]
    relu6_00 = tl.minimum(tl.maximum(a00, 0.0), 6.0)
    relu6_01 = tl.minimum(tl.maximum(a01, 0.0), 6.0)
    relu6_10 = tl.minimum(tl.maximum(a10, 0.0), 6.0)
    relu6_11 = tl.minimum(tl.maximum(a11, 0.0), 6.0)

    hs00 = sv00 * (relu6_00 / 6.0)
    hs01 = sv01 * (relu6_01 / 6.0)
    hs10 = sv10 * (relu6_10 / 6.0)
    hs11 = sv11 * (relu6_11 / 6.0)

    # max pooling across 4 values
    m0 = tl.where(hs00 > hs01, hs00, hs01)
    m1 = tl.where(hs10 > hs11, hs10, hs11)
    m = tl.where(m0 > m1, m0, m1)

    # Mish activation: m * tanh(softplus(m))
    # softplus(x) = log(1 + exp(x)) but for numerical stability clip for large x
    thresh = 20.0
    # use log(1+exp(m)) for numerical stability; avoid using log1p (not available)
    sp = tl.where(m > thresh, m, tl.log(1.0 + tl.exp(m)))
    # tanh(sp) = 2*sigmoid(2*sp) - 1
    sig = 1.0 / (1.0 + tl.exp(-2.0 * sp))
    tanh_sp = 2.0 * sig - 1.0
    mish = m * tanh_sp

    # store result back to flattened output: ((n*C + c) * pooled_H * pooled_W) + idx_in_plane
    out_base = ((n * C) + c) * (pooled_H * pooled_W)
    out_idx = out_base + idx_in_plane
    tl.store(out_ptr + out_idx, mish, mask=mask_pos)


class ModelNew(nn.Module):
    """
    Optimized Model that keeps PyTorch's Conv2d (cuDNN) and fuses:
      - subtract scalar
      - HardSwish
      - 2x2 MaxPool (stride 2)
      - Mish
    into a single Triton kernel for pool size 2. Falls back to PyTorch ops for other pool sizes.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super(ModelNew, self).__init__()
        # keep PyTorch conv (fast cuDNN)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = float(subtract_value)
        self.pool_kernel_size = pool_kernel_size
        # this fused kernel is implemented for 2x2 pooling with stride 2
        self.fuse_pool = (self.pool_kernel_size == 2)

    def forward(self, x: torch.Tensor):
        # perform conv with PyTorch/cuDNN
        conv_out = self.conv(x)
        if not self.fuse_pool:
            # fallback to original PyTorch ops if pool size != 2
            x = conv_out - self.subtract_value
            x = torch.nn.functional.hardswish(x)
            x = nn.functional.max_pool2d(x, kernel_size=self.pool_kernel_size)
            x = torch.nn.functional.mish(x)
            return x

        # fused path for pool size 2
        inp = conv_out.contiguous()
        N, C, H, W = inp.shape
        pooled_H = H // 2
        pooled_W = W // 2

        out = torch.empty((N, C, pooled_H, pooled_W), device=inp.device, dtype=inp.dtype)
        numel = N * C * pooled_H * pooled_W
        P = pooled_H * pooled_W

        # grid depends on the BLOCK chosen by autotune; use meta in lambda to compute tiles
        grid = lambda meta: (N * C * ((P + meta["BLOCK"] - 1) // meta["BLOCK"]),)

        # launch the fused Triton kernel (autotune will pick best BLOCK/warps)
        _fused_sub_hswish_pool_mish_kernel[grid](
            inp, out,
            N, C, H, W,
            pooled_H, pooled_W,
            float(self.subtract_value),
            numel
        )
        return out