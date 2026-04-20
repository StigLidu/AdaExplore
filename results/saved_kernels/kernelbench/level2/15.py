import torch
import torch.nn as nn
import triton
import triton.language as tl

# Fused Triton kernel:
# For each (n, c) slice (size S = D*H*W), we:
#   1) Compute mean_x = mean(x) over spatial elements.
#   2) Compute scale = gamma / sqrt(running_var + eps) (or gamma*invstd).
#   3) Write out: out = scale * (x - mean_x)
#
# This leverages the algebra:
#   bn = gamma * (x - rm) * invstd + beta
#   mean_bn = gamma * (mean_x - rm) * invstd + beta
#   bn - mean_bn = gamma * invstd * (x - mean_x)
# so we only need mean_x and scale, reducing arithmetic and memory ops.
@triton.jit
def _bn_sub_mean_kernel_v2(
    inp_ptr,           # pointer to input tensor (flattened)
    out_ptr,           # pointer to output tensor (flattened)
    N,                 # batch size
    C,                 # channels
    S,                 # spatial size D*H*W
    gamma_ptr,         # per-channel gamma (weight) length C
    running_var_ptr,   # per-channel running var length C (or batch var passed in)
    eps,               # epsilon scalar
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)          # one program per (n, c)
    n = pid // C
    c = pid % C
    base = (n * C + c) * S         # base offset for this (n,c) slice

    offs = tl.arange(0, BLOCK)     # offsets handled by this program per iteration

    # Load per-channel parameters
    gamma = tl.load(gamma_ptr + c)
    rv = tl.load(running_var_ptr + c)
    invstd = 1.0 / tl.sqrt(rv + eps)
    scale = gamma * invstd

    # First pass: compute sum_x over spatial, to get mean_x
    total = 0.0
    s = 0
    while s < S:
        idx = offs + s
        mask = idx < S
        ptr = inp_ptr + base + idx
        x = tl.load(ptr, mask=mask, other=0.0)
        # loaded out-of-bounds lanes are zero because of other=0.0
        total = total + tl.sum(x)
        s += BLOCK

    mean_x = total / S

    # Second pass: write out = scale * (x - mean_x)
    s = 0
    while s < S:
        idx = offs + s
        mask = idx < S
        ptr_in = inp_ptr + base + idx
        x = tl.load(ptr_in, mask=mask, other=0.0)
        out_vals = scale * (x - mean_x)
        ptr_out = out_ptr + base + idx
        tl.store(ptr_out, out_vals, mask=mask)
        s += BLOCK


def _apply_bn_and_sub_mean_triton_v2(x: torch.Tensor, bn_module: nn.BatchNorm3d):
    """
    Applies BatchNorm (using bn_module params: weight, running_var, eps)
    and subtracts the spatial mean per (batch, channel) using a fused Triton kernel.

    The kernel uses the algebraic simplification:
      bn - mean(bn) = gamma * invstd * (x - mean_x)
    which avoids extra arithmetic and memory loads.
    """
    assert x.is_cuda, "Input must be on CUDA"
    x = x.contiguous()
    N, C, D, H, W = x.shape
    S = D * H * W

    device = x.device
    dtype = x.dtype

    # Prepare per-channel gamma and running_var (or batch stats if in training)
    if bn_module.affine:
        gamma = bn_module.weight.detach().to(device=device, dtype=dtype).contiguous()
    else:
        gamma = torch.ones(C, device=device, dtype=dtype)

    eps = float(bn_module.eps)

    if bn_module.training:
        # When training, use batch statistics (mean/var over N and spatial dims)
        # The kernel needs per-channel variance (not running); compute here on GPU.
        # Use unbiased=False to match BatchNorm behavior.
        batch_var = x.var(dim=(0, 2, 3, 4), unbiased=False).detach().to(device=device, dtype=dtype).contiguous()
        var_ptr = batch_var
    else:
        var_ptr = bn_module.running_var.detach().to(device=device, dtype=dtype).contiguous()

    out = torch.empty_like(x)

    # Grid: one program per (n, c)
    grid = (N * C,)

    # BLOCK size: chosen to balance occupancy and #iterations.
    # 4096 processes 4k spatial elements per loop iteration; S=16*32*32=16384 -> 4 iterations.
    BLOCK = 4096

    _bn_sub_mean_kernel_v2[grid](
        x, out,
        N, C, S,
        gamma, var_ptr,
        eps,
        BLOCK=BLOCK
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized Model: we keep PyTorch's ConvTranspose3d for the transpose convolution,
    but replace the BatchNorm3d + spatial-mean-subtraction with a fused Triton kernel
    that computes the spatial mean per (batch,channel) and writes the final output
    using the algebraic simplification to minimize memory loads and arithmetic.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels,
                                                 kernel_size, stride=stride,
                                                 padding=padding, bias=bias)
        # Keep a BatchNorm3d module to hold running stats and affine params.
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = _apply_bn_and_sub_mean_triton_v2(x, self.batch_norm)
        return x


# Re-create the get_inputs and get_init_inputs functions for compatibility/testing
batch_size = 16
in_channels = 16
out_channels = 32
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1

def get_inputs():
    # Return a CUDA tensor as the kernels expect CUDA inputs.
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]