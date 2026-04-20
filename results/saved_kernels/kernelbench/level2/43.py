import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configurations for the max-pool kernel.
# Use BLOCK sizes aligned to typical cache/coalescing boundaries and expose ROWS_PER_PROG
# so the autotuner can pick how many rows each program handles. Added larger/wider
# configurations for the A6000 memory-bound regime.
AUTOTUNE_MAXPOOL = [
    triton.Config({"BLOCK": 64,  "ROWS_PER_PROG": 8},  num_warps=4,  num_stages=2),
    triton.Config({"BLOCK": 128, "ROWS_PER_PROG": 8},  num_warps=4,  num_stages=2),
    triton.Config({"BLOCK": 256, "ROWS_PER_PROG": 4},  num_warps=8,  num_stages=2),
    triton.Config({"BLOCK": 256, "ROWS_PER_PROG": 4},  num_warps=8,  num_stages=3),
    triton.Config({"BLOCK": 512, "ROWS_PER_PROG": 2},  num_warps=16, num_stages=3),
    triton.Config({"BLOCK": 512, "ROWS_PER_PROG": 4},  num_warps=16, num_stages=3),
]

@triton.autotune(
    configs=AUTOTUNE_MAXPOOL,
    key=["B", "C", "out_D", "out_H", "out_W"]
)
@triton.jit
def _maxpool3d_kernel(
    inp,              # input tensor pointer (N, C, D, H, W)
    out,              # output tensor pointer (N, C, out_D, out_H, out_W)
    B, C, D, H, W,    # input dims
    out_D, out_H, out_W,  # output spatial dims
    n_rows,           # = B * C * out_D * out_H
    BLOCK: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr
):
    # We tile the row dimension so each Triton program handles multiple (b, c, od, oh) rows.
    pid0 = tl.program_id(0)
    col_block = tl.program_id(1)

    # Each program processes a contiguous block of output width positions
    offs = col_block * BLOCK + tl.arange(0, BLOCK)
    mask_w = offs < out_W

    # Iterate over multiple rows per program to amortize launch overhead.
    for r in range(ROWS_PER_PROG):
        row = pid0 * ROWS_PER_PROG + r
        if row < n_rows:
            # decode row -> (b, c, od, oh)
            tmp = row
            oh = tmp % out_H
            tmp = tmp // out_H
            od = tmp % out_D
            tmp = tmp // out_D
            c = tmp % C
            b = tmp // C

            # input indices (top-left-front corner of the 2x2x2 pooling window)
            iw = offs * 2  # stride = 2
            ih = oh * 2
            id0 = od * 2

            # compute a base linear index for input at (b, c, id0, ih, 0)
            # linear_input_base = ((((b * C + c) * D + id0) * H + ih) * W)
            b_c = b * C + c
            linear_input_base = (((b_c * D + id0) * H + ih) * W)

            # Prepare masks for boundary checks and compute corner masks (vectorized).
            # width checks: iw and iw+1 must be < W (vector)
            mask_iw0 = iw < W
            mask_iw1 = iw + 1 < W

            # height/depth scalar conditions (broadcasted)
            has_h0 = (ih < H)
            has_h1 = (ih + 1) < H
            has_d0 = (id0 < D)
            has_d1 = (id0 + 1) < D

            # compute addresses for the 8 corners (vectorized over offs) in grouped form
            base = linear_input_base + iw  # id0, ih0
            H_stride = W
            D_stride = H * W

            # Grouped plane bases:
            base_id0_ih0 = base                         # id0, ih0
            base_id0_ih1 = base + H_stride              # id0, ih1
            base_id1_ih0 = base + D_stride              # id1, ih0
            base_id1_ih1 = base + D_stride + H_stride   # id1, ih1

            # load each plane's two contiguous width elements (iw, iw+1) with masks computed once.
            neg_inf = -1e9

            mask_row_id0_ih0 = mask_w & has_d0 & has_h0
            v00 = tl.load(inp + base_id0_ih0,     mask=mask_row_id0_ih0 & mask_iw0, other=neg_inf)
            v01 = tl.load(inp + base_id0_ih0 + 1, mask=mask_row_id0_ih0 & mask_iw1, other=neg_inf)

            mask_row_id0_ih1 = mask_w & has_d0 & has_h1
            v10 = tl.load(inp + base_id0_ih1,     mask=mask_row_id0_ih1 & mask_iw0, other=neg_inf)
            v11 = tl.load(inp + base_id0_ih1 + 1, mask=mask_row_id0_ih1 & mask_iw1, other=neg_inf)

            mask_row_id1_ih0 = mask_w & has_d1 & has_h0
            v20 = tl.load(inp + base_id1_ih0,     mask=mask_row_id1_ih0 & mask_iw0, other=neg_inf)
            v21 = tl.load(inp + base_id1_ih0 + 1, mask=mask_row_id1_ih0 & mask_iw1, other=neg_inf)

            mask_row_id1_ih1 = mask_w & has_d1 & has_h1
            v30 = tl.load(inp + base_id1_ih1,     mask=mask_row_id1_ih1 & mask_iw0, other=neg_inf)
            v31 = tl.load(inp + base_id1_ih1 + 1, mask=mask_row_id1_ih1 & mask_iw1, other=neg_inf)

            # map to canonical corner names
            v000 = v00
            v001 = v01
            v010 = v10
            v011 = v11
            v100 = v20
            v101 = v21
            v110 = v30
            v111 = v31

            # compute max across the 8 values
            m01 = tl.maximum(v000, v001)
            m23 = tl.maximum(v010, v011)
            m45 = tl.maximum(v100, v101)
            m67 = tl.maximum(v110, v111)

            m0123 = tl.maximum(m01, m23)
            m4567 = tl.maximum(m45, m67)
            m = tl.maximum(m0123, m4567)

            # Write out to output tensor
            out_base = (((b_c * out_D + od) * out_H + oh) * out_W)
            out_addr = out + out_base + offs
            tl.store(out_addr, m, mask=mask_w)


def triton_maxpool3d(x: torch.Tensor, kernel_size=2, stride=2):
    """
    Triton-based 3D max pool for kernel_size=2, stride=2.
    Accepts x of shape (B, C, D, H, W) and returns pooled tensor of shape (B, C, D//2, H//2, W//2).
    """
    assert x.is_cuda, "Input must be on CUDA"
    assert x.dtype == torch.float32, "Only float32 supported"
    assert kernel_size == 2 and stride == 2, "This Triton kernel supports only kernel_size=2 and stride=2"

    x = x.contiguous()
    B, C, D, H, W = x.shape
    out_D = D // 2
    out_H = H // 2
    out_W = W // 2

    out = torch.empty((B, C, out_D, out_H, out_W), device=x.device, dtype=x.dtype)

    n_rows = B * C * out_D * out_H
    # grid: (rows_groups, num_w_tiles_along_W) -- we tile rows so each program handles multiple rows
    grid = lambda meta: ((n_rows + meta["ROWS_PER_PROG"] - 1) // meta["ROWS_PER_PROG"], (out_W + meta["BLOCK"] - 1) // meta["BLOCK"])

    _maxpool3d_kernel[grid](
        x, out,
        B, C, D, H, W,
        out_D, out_H, out_W,
        n_rows
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that uses a Triton-based 3D max-pool kernel for the MaxPool3d operation,
    while keeping nn.Conv3d for the convolution and using PyTorch for logsumexp + ReLU.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        # x: (B, in_channels, D, H, W)
        x = self.conv(x)                       # (B, out_channels, D, H, W)
        x = triton_maxpool3d(x, kernel_size=2, stride=2)  # (B, out_channels, D//2, H//2, W//2)
        x = torch.logsumexp(x, dim=1, keepdim=True)      # (B, 1, D//2, H//2, W//2)
        x = torch.relu(x)
        return x