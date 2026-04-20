import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotuning: try a few block sizes optimized for A6000
AUTOTUNE_CONFIGS = [
    # Larger search space tuned for Ampere (A6000). Keep BLOCK_C=8 (matches C==8).
    triton.Config({"BLOCK_OC": 32, "BLOCK_HW": 256, "BLOCK_C": 8}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_OC": 64, "BLOCK_HW": 256, "BLOCK_C": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_OC": 128, "BLOCK_HW": 256, "BLOCK_C": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_OC": 32, "BLOCK_HW": 512, "BLOCK_C": 8}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_OC": 64, "BLOCK_HW": 512, "BLOCK_C": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_OC": 128, "BLOCK_HW": 512, "BLOCK_C": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_OC": 32, "BLOCK_HW": 1024, "BLOCK_C": 8}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_OC": 64, "BLOCK_HW": 1024, "BLOCK_C": 8}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['N', 'C', 'H', 'W', 'OC', 'K'])
@triton.jit
def conv2d_fused_kernel(
    x_ptr,        # input tensor pointer (N, C, H, W)
    w_ptr,        # weight pointer (OC, C, K, K)
    b_ptr,        # bias pointer (OC,)
    out_ptr,      # output pointer (N, OC, H_out, W_out)
    N, C, H, W, OC, HW_out, divisor,
    K: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # program ids
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    H_out = H - K + 1
    W_out = W - K + 1

    # oc and hw indices handled by this program
    oc_ids = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)             # [BLOCK_OC]
    hw_ids = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)             # [BLOCK_HW]

    mask_oc = oc_ids < OC
    mask_hw = hw_ids < (H_out * W_out)

    # compute spatial coords for hw positions
    h_idx = hw_ids // W_out  # [BLOCK_HW]
    w_idx = hw_ids % W_out   # [BLOCK_HW]

    # accumulator
    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    # iterate over input channels in blocks of BLOCK_C to improve data reuse
    for c_start in range(0, C, BLOCK_C):
        # vector of channel indices in this block: [BLOCK_C]
        ic_range = tl.arange(0, BLOCK_C)
        ic = c_start + ic_range  # [BLOCK_C]
        valid_ic = ic < C  # [BLOCK_C] boolean mask

        # for each kernel spatial position load a [BLOCK_C, BLOCK_HW] input tile and a
        # [BLOCK_OC, BLOCK_C] weight tile then do a broadcasted multiply and reduce over C.
        for kh in range(0, K):
            for kw in range(0, K):
                # input positions for this kernel offset
                in_h = h_idx + kh  # [BLOCK_HW]
                in_w = w_idx + kw  # [BLOCK_HW]

                # compute input offsets for every (ic, hw): shape [BLOCK_C, BLOCK_HW]
                # ((pid_n * C + ic) * H * W)[:, None] + in_h[None, :] * W + in_w[None, :]
                in_base = (pid_n * C + ic)[:, None] * H * W  # [BLOCK_C, 1]
                in_offs = in_base + in_h[None, :] * W + in_w[None, :]  # [BLOCK_C, BLOCK_HW]

                mask_in = valid_ic[:, None] & mask_hw[None, :]  # [BLOCK_C, BLOCK_HW]
                inp = tl.load(x_ptr + in_offs, mask=mask_in, other=0.0)  # [BLOCK_C, BLOCK_HW]

                # weight offsets for this (oc, ic, kh, kw)
                # base index for (oc, ic) pair -> add kernel offset inside K*K
                w_base = (oc_ids[:, None] * C + ic[None, :]) * (K * K)  # [BLOCK_OC, BLOCK_C]
                w_offs = w_base + kh * K + kw  # [BLOCK_OC, BLOCK_C]
                mask_w = mask_oc[:, None] & valid_ic[None, :]  # [BLOCK_OC, BLOCK_C]
                w_vals = tl.load(w_ptr + w_offs, mask=mask_w, other=0.0)  # [BLOCK_OC, BLOCK_C]

                # multiply-accumulate: w_vals [BLOCK_OC, BLOCK_C], inp [BLOCK_C, BLOCK_HW]
                # broadcast to [BLOCK_OC, BLOCK_C, BLOCK_HW] then reduce over C
                prod = w_vals[:, :, None] * inp[None, :, :]  # [BLOCK_OC, BLOCK_C, BLOCK_HW]
                acc += tl.sum(prod, 1)  # sum over channel dim -> [BLOCK_OC, BLOCK_HW]

    # add bias
    b_vals = tl.load(b_ptr + oc_ids, mask=mask_oc, other=0.0)  # [BLOCK_OC]
    acc += b_vals[:, None]

    # divide by scalar divisor
    inv_div = 1.0 / divisor
    acc = acc * inv_div

    # leaky relu with negative_slope=0.01
    pos = tl.maximum(acc, 0.0)
    neg = tl.minimum(acc, 0.0) * 0.01
    out_vals = pos + neg  # [BLOCK_OC, BLOCK_HW]

    # compute output offsets and store
    out_base = pid_n * OC * HW_out
    out_offs = out_base + oc_ids[:, None] * HW_out + hw_ids[None, :]  # [BLOCK_OC, BLOCK_HW]
    mask_out = mask_oc[:, None] & mask_hw[None, :]
    tl.store(out_ptr + out_offs, out_vals, mask=mask_out)


def triton_conv2d_fused(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, divisor: float, K: int):
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    assert x.dtype == torch.float32 and weight.dtype == torch.float32 and bias.dtype == torch.float32

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    N, C, H, W = x.shape
    OC, C_w, K_w, K_w2 = weight.shape
    assert C_w == C and K_w == K and K_w2 == K, "Weight shape mismatch."

    H_out = H - K + 1
    W_out = W - K + 1
    HW_out = H_out * W_out

    out = torch.empty((N, OC, H_out, W_out), device=x.device, dtype=x.dtype)

    def grid(meta):
        BLOCK_OC = meta['BLOCK_OC']
        BLOCK_HW = meta['BLOCK_HW']
        return (N, (OC + BLOCK_OC - 1) // BLOCK_OC, (HW_out + BLOCK_HW - 1) // BLOCK_HW)

    conv2d_fused_kernel[grid](
        x, weight, bias, out,
        N, C, H, W, OC, HW_out, float(divisor),
        K=K
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that fuses Conv2d (no padding, stride=1) + divide by constant + leaky_relu
    into a Triton kernel for improved performance on GPU.
    """

    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super(ModelNew, self).__init__()
        # Keep a regular conv module for parameter storage
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = float(divisor)
        self.kernel_size = kernel_size

    def forward(self, x):
        # Ensure params are on same device as input
        if x.device != self.conv.weight.device:
            self.conv.to(x.device)

        return triton_conv2d_fused(x, self.conv.weight, self.conv.bias, self.divisor, self.kernel_size)


# Keep input helpers consistent with original
batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3
divisor = 2


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor]