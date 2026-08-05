import torch
import torch.nn as nn
import triton
import triton.language as tl

# Fused Triton kernel that performs 2x2x2 max-pooling (stride=2),
# followed by channel-wise logsumexp and ReLU.
# Input layout: (B, C, D, H, W) flattened
# Output layout: (B, 1, Dp, Hp, Wp) flattened
@triton.jit
def _fused_pool_lse_relu_kernel(
    inp_ptr,            # pointer to input tensor (B, C, D, H, W) flattened
    out_ptr,            # pointer to output tensor (B, 1, Dp, Hp, Wp) flattened
    B, C,               # batch, channels
    D, H, W,            # input spatial dims (unpooled)
    Dp, Hp, Wp,         # pooled spatial dims (Dp = D//2, ...)
    stride_c,           # stride to move one channel: D*H*W
    stride_n,           # stride to move one batch: C*D*H*W
    NPOS,               # total number of pooled positions = B * Dp * Hp * Wp
    NEG_INF,            # very negative constant for masked loads
    BLOCK_POS: tl.constexpr,  # number of pooled positions per program (constexpr)
    BLOCK_C: tl.constexpr,    # channel block size for inner reduction (constexpr)
):
    pid = tl.program_id(0)
    tile_start = pid * BLOCK_POS
    offs = tl.arange(0, BLOCK_POS)
    pos = tile_start + offs
    mask_pos = pos < NPOS  # active pooled positions

    # pooled positions per channel
    CH_SZ_POOLED = Dp * Hp * Wp
    # compute batch index and pooled-position-in-batch
    n = pos // CH_SZ_POOLED
    pos_in_pooled = pos - n * CH_SZ_POOLED

    # decompose pooled index into pd, ph, pw
    plane = Hp * Wp
    pd = pos_in_pooled // plane
    rem = pos_in_pooled - pd * plane
    ph = rem // Wp
    pw = rem - ph * Wp

    # base coordinate in unpooled tensor (top-left-front corner of 2x2x2 window)
    d_base = pd * 2
    h_base = ph * 2
    w_base = pw * 2
    pos_un_base = d_base * (H * W) + h_base * W + w_base

    # base pointer for each pooled position (start of batch + pos_un_base)
    base_ptr = n * stride_n + pos_un_base  # vector length BLOCK_POS

    # offsets for the 8 elements inside the 2x2x2 pooling window
    off000 = 0
    off001 = 1
    off010 = W
    off011 = W + 1
    off100 = H * W
    off101 = H * W + 1
    off110 = H * W + W
    off111 = H * W + W + 1

    # Initialize online LSE accumulators: m (max) and s (sum of exp(vals - m))
    m = tl.full((BLOCK_POS,), NEG_INF, dtype=tl.float32)
    s = tl.zeros((BLOCK_POS,), dtype=tl.float32)

    # Iterate channels in blocks to control register pressure / occupancy
    for c0 in range(0, C, BLOCK_C):
        for c_inner in range(0, BLOCK_C):
            ch = c0 + c_inner
            ch_mask = ch < C  # scalar boolean
            # pointer to beginning of this channel's window for each pooled position
            ch_ptr = base_ptr + ch * stride_c
            active = mask_pos & ch_mask  # lanes where this channel & pooled-pos are valid

            # load 8 values of the 2x2x2 window and compute per-channel pooled max
            v0 = tl.load(inp_ptr + ch_ptr + off000, mask=active, other=NEG_INF)
            v1 = tl.load(inp_ptr + ch_ptr + off001, mask=active, other=NEG_INF)
            v2 = tl.load(inp_ptr + ch_ptr + off010, mask=active, other=NEG_INF)
            v3 = tl.load(inp_ptr + ch_ptr + off011, mask=active, other=NEG_INF)
            v4 = tl.load(inp_ptr + ch_ptr + off100, mask=active, other=NEG_INF)
            v5 = tl.load(inp_ptr + ch_ptr + off101, mask=active, other=NEG_INF)
            v6 = tl.load(inp_ptr + ch_ptr + off110, mask=active, other=NEG_INF)
            v7 = tl.load(inp_ptr + ch_ptr + off111, mask=active, other=NEG_INF)

            vm = v0
            vm = tl.maximum(vm, v1)
            vm = tl.maximum(vm, v2)
            vm = tl.maximum(vm, v3)
            vm = tl.maximum(vm, v4)
            vm = tl.maximum(vm, v5)
            vm = tl.maximum(vm, v6)
            vm = tl.maximum(vm, v7)

            # To avoid invalid lanes interfering, set vm_eff = m for inactive lanes
            vm_eff = tl.where(active, vm, m)

            # Online numerically-stable update of log-sum-exp accumulators:
            # If vm_eff > m: s_new = 1 + s * exp(m - vm_eff)
            # else: s_new = s + exp(vm_eff - m)
            greater = vm_eff > m
            exp_m_vm = tl.exp(m - vm_eff)
            exp_vm_m = tl.exp(vm_eff - m)
            s_new = tl.where(greater, 1.0 + s * exp_m_vm, s + exp_vm_m)

            # commit updates only on active lanes
            s = tl.where(active, s_new, s)
            m = tl.where(active, tl.maximum(m, vm), m)

    # finalize logsumexp and apply ReLU
    lse = m + tl.log(s)
    zero = tl.zeros((BLOCK_POS,), dtype=tl.float32)
    out_val = tl.maximum(lse, zero)

    # compute output pointers and store (output layout: (B,1,Dp,Hp,Wp))
    out_stride_n = CH_SZ_POOLED
    out_ptrs = n * out_stride_n + pos_in_pooled
    tl.store(out_ptr + out_ptrs, out_val, mask=mask_pos)


def triton_fused_pool_lse_relu(x: torch.Tensor):
    """
    Wrapper for the Triton kernel that fuses 2x2x2 maxpool (stride=2) and channel logsumexp+ReLU.
    Input: x: (B, C, D, H, W), cuda, float32
    Output: (B, 1, Dp, Hp, Wp), cuda, float32
    """
    assert x.is_cuda, "Input must be on CUDA"
    assert x.dtype == torch.float32, "Input must be float32"
    x = x.contiguous()
    B, C, D, H, W = x.shape

    Dp = D // 2
    Hp = H // 2
    Wp = W // 2

    out = torch.empty((B, 1, Dp, Hp, Wp), device=x.device, dtype=x.dtype)

    NEG_INF = float(-1e20)

    # Tunable parameters:
    # Increase BLOCK_POS to improve coalesced loads and reduce launch overhead.
    # Choose BLOCK_C to balance register pressure. For C=64, BLOCK_C=16 gives 4 iterations.
    BLOCK_POS = 2048  # must be multiple of 32
    BLOCK_C = 16

    stride_c = D * H * W
    stride_n = C * stride_c
    NPOS = B * Dp * Hp * Wp

    grid = ((NPOS + BLOCK_POS - 1) // BLOCK_POS,)

    _fused_pool_lse_relu_kernel[grid](
        x,
        out,
        B, C,
        D, H, W,
        Dp, Hp, Wp,
        stride_c, stride_n, NPOS,
        NEG_INF,
        BLOCK_POS=BLOCK_POS,
        BLOCK_C=BLOCK_C,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model: use PyTorch Conv3d, and a fused Triton kernel that
    performs 2x2x2 max-pool + channel logsumexp + ReLU in one pass.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        # keep convolution in PyTorch (highly optimized)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        # fused pooling + reduction + activation
        x = triton_fused_pool_lse_relu(x)
        return x


# Keep helpers for compatibility
batch_size = 4
in_channels = 32
out_channels = 64
depth, height, width = 32, 128, 128
kernel_size = 3
stride = 1
padding = 1

def get_inputs():
    # return a CUDA tensor ready for the Triton kernel
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda().float()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]