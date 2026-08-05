import torch
import torch.nn as nn
import triton
import triton.language as tl

# Triton kernel: Compute Mish (x * tanh(softplus(x))) over a block of elements for a specific channel,
# accumulate partial sums (sum and sum of squares) into per-channel global accumulators using atomic adds.
@triton.jit
def _mish_reduce_kernel(
    x_ptr,        # pointer to conv input (N*C*H*W)
    sum_ptr,      # pointer to per-channel sum (C,)
    sumsq_ptr,    # pointer to per-channel sum of squares (C,)
    m_ptr,         # pointer to temp Mish buffer (N*C*H*W) stored as fp16
    elems_per_ch, # number of elements per channel (N*H*W)
    C,            # number of channels
    H,            # height
    W,            # width
    BLOCK: tl.constexpr,
):
    c = tl.program_id(0)           # channel index
    block_id = tl.program_id(1)    # block index over elems_per_ch

    start = block_id * BLOCK
    offs = start + tl.arange(0, BLOCK)
    mask = offs < elems_per_ch

    HW = H * W
    CHW = C * HW

    # compute n and hw coords: n = idx // (H*W), hw = idx % (H*W)
    n = offs // HW
    hw = offs - n * HW

    # compute memory offsets for elements: offset = n*(C*H*W) + c*(H*W) + hw
    offsets = n * CHW + c * HW + hw

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # softplus: use a numerically stable variant
    # clamp to avoid overflow in exp
    x_clamped = tl.minimum(x, 20.0)
    exp_x = tl.exp(x_clamped)
    sp = tl.where(x > 20.0, x, tl.log(1.0 + exp_x))

    # tanh(sp) computed as (1 - exp(-2sp)) / (1 + exp(-2sp))
    neg2sp = -2.0 * sp
    exp_neg2sp = tl.exp(neg2sp)
    tanh_sp = (1.0 - exp_neg2sp) / (1.0 + exp_neg2sp)

    m = x * tanh_sp  # mish

    # compute block-wise sums
    s = tl.sum(m, axis=0)
    s2 = tl.sum(m * m, axis=0)

    # Atomic add partial sums to global per-channel accumulators
    tl.atomic_add(sum_ptr + c, s)
    tl.atomic_add(sumsq_ptr + c, s2)

    # store per-element Mish into temporary buffer in fp16 to avoid recomputing later
    m_fp16 = tl.cast(m, tl.float16)
    tl.store(m_ptr + offsets, m_fp16, mask=mask)


# Triton kernel: compute Mish and then apply per-channel affine transform
# out = (x * tanh(softplus(x))) * scale[c] + bias[c]
@triton.jit
def _mish_bn_apply_kernel(
    m_ptr,        # pointer to stored Mish activations (N*C*H*W) in fp16
    out_ptr,      # pointer to output (N*C*H*W)
    scale_ptr,    # per-channel scale (C,)
    bias_ptr,     # per-channel bias_term (C,)
    elems_per_ch, # number of elements per channel (N*H*W)
    C,            # number of channels
    H,            # height
    W,            # width
    BLOCK: tl.constexpr,
):
    c = tl.program_id(0)           # channel index
    block_id = tl.program_id(1)    # block index over elems_per_ch

    start = block_id * BLOCK
    offs = start + tl.arange(0, BLOCK)
    mask = offs < elems_per_ch

    HW = H * W
    CHW = C * HW

    n = offs // HW
    hw = offs - n * HW

    offsets = n * CHW + c * HW + hw

    # load stored Mish (fp16) and cast back to fp32
    m_fp16 = tl.load(m_ptr + offsets, mask=mask, other=0.0)
    m = tl.cast(m_fp16, tl.float32)

    # load per-channel scale and bias
    scale = tl.load(scale_ptr + c)
    bias = tl.load(bias_ptr + c)

    out = m * scale + bias

    tl.store(out_ptr + offsets, out, mask=mask)


class ModelNew(nn.Module):
    """
    Optimized Model:
      - Keep PyTorch Conv2d and BatchNorm2d modules for parameter/buffer management.
      - Use Triton kernels to:
        1) compute Mish and simultaneously accumulate per-channel sum and sum-of-squares (single pass).
        2) compute batch mean/var on the accumulated sums (on CPU/GPU via PyTorch ops).
        3) apply BatchNorm affine (scale + bias) fused with Mish in a second Triton pass.
      This reduces memory traffic by removing an intermediate full-tensor read/write for activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)

        # block size for Triton kernels; tuneable
        # use a smaller block to reduce register pressure and improve occupancy on Ampere
        self._BLOCK = 512

    def forward(self, x):
        # x: (N, C_in, H, W)
        x = self.conv(x)  # use PyTorch conv

        # If not on CUDA, fallback to PyTorch implementation for correctness
        if not x.is_cuda:
            x = x.contiguous()
            x = x * torch.tanh(torch.nn.functional.softplus(x))
            x = self.bn(x)
            return x

        x = x.contiguous()
        N, C, H, W = x.shape
        elems_per_ch = N * H * W

        # prepare per-channel accumulators on device
        device = x.device
        dtype = x.dtype
        sum_buf = torch.zeros(C, device=device, dtype=dtype)
        sumsq_buf = torch.zeros(C, device=device, dtype=dtype)
        # temporary buffer to store per-element Mish activations in fp16 to avoid recomputing
        m_buf = torch.empty_like(x, dtype=torch.float16, device=device)

        # Launch reduction kernel to compute per-channel sums and sumsq of Mish(x)
        num_blocks = (elems_per_ch + self._BLOCK - 1) // self._BLOCK
        grid = (C, num_blocks)
        _mish_reduce_kernel[grid](
            x, sum_buf, sumsq_buf, m_buf,
            elems_per_ch, C, H, W,
            BLOCK=self._BLOCK
        )

        # compute mean and var from accumulators
        # sum and sumsq are on GPU; do computations on GPU to avoid device sync/copies
        mean = sum_buf / elems_per_ch  # shape (C,)
        ex2 = sumsq_buf / elems_per_ch
        var = ex2 - mean * mean
        # numerical noise can make var slightly negative; clamp
        var = torch.clamp(var, min=0.0)

        # Update running stats if training
        if self.bn.training:
            with torch.no_grad():
                rm = self.bn.running_mean
                rv = self.bn.running_var
                momentum = self.bn.momentum
                # Move mean/var to the buffers' device if necessary
                if mean.device != rm.device:
                    mean_dev = mean.to(rm.device)
                    var_dev = var.to(rv.device)
                else:
                    mean_dev = mean
                    var_dev = var
                rm.mul_(1.0 - momentum).add_(momentum * mean_dev)
                rv.mul_(1.0 - momentum).add_(momentum * var_dev)

        # Prepare affine parameters: scale = weight * invstd, bias_term = bias - mean*scale
        device = mean.device
        if self.bn.weight is not None:
            weight = self.bn.weight.to(device)
        else:
            weight = torch.ones_like(mean, device=device)
        if self.bn.bias is not None:
            bias = self.bn.bias.to(device)
        else:
            bias = torch.zeros_like(mean, device=device)

        invstd = 1.0 / torch.sqrt(var + self.bn.eps)
        scale = weight * invstd
        bias_term = bias - mean * scale

        # ensure contiguous on device
        scale = scale.contiguous()
        bias_term = bias_term.contiguous()

        # Allocate output tensor
        out = torch.empty_like(x)

        # Launch apply kernel: load precomputed Mish from m_buf and apply per-channel affine, writing to out
        _mish_bn_apply_kernel[grid](
            m_buf, out, scale, bias_term,
            elems_per_ch, C, H, W,
            BLOCK=self._BLOCK
        )

        return out