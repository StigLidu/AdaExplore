import torch
import torch.nn as nn
import triton
import triton.language as tl

# Fused single-kernel reduction: scan the spatial dimension per (n,c) program,
# accumulate scaled sums directly, and apply batchnorm with running stats.
# This removes the intermediate partials buffer and extra kernel launch.
# Choose a moderately large tile for good memory throughput on Ampere.
FUSED_BLOCK = 512

@triton.jit
def _mean_bn_pool_kernel(
    x_ptr,           # pointer to input tensor (N*C*L elements, contiguous with layout [N, C, L])
    out_ptr,         # pointer to output tensor (N*C elements)
    gamma_ptr,       # pointer to batchnorm weight (C)
    beta_ptr,        # pointer to batchnorm bias (C)
    rm_ptr,          # pointer to running_mean (C)
    rv_ptr,          # pointer to running_var (C)
    scale,           # scalar scale_factor (float)
    eps,             # scalar eps (float)
    N,               # number of batches
    C,               # number of channels
    L,               # spatial size (D*H*W)
    BLOCK: tl.constexpr,  # block size for inner reduction (constexpr)
):
    pid = tl.program_id(0)  # one program per (n,c)
    total = N * C
    if pid >= total:
        return

    n = pid // C
    c = pid % C

    # base pointer offset for this (n, c) over flattened layout where
    # entries are stored as: ((n * C + c) * L) + offset_in_spatial
    base = (n * C + c) * L

    # accumulator for the scaled sum across spatial elements
    acc = 0.0

    offs = tl.arange(0, BLOCK)
    # Loop over the spatial dimension in chunks of BLOCK
    for off in range(0, L, BLOCK):
        idx = off + offs
        mask = idx < L
        vals = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        # multiply by scale during accumulation (minor optimization)
        acc += tl.sum(vals * scale)

    # compute mean (acc already includes scaling)
    mean_x = acc / L

    # load batchnorm parameters for this channel
    gamma = tl.load(gamma_ptr + c)
    beta = tl.load(beta_ptr + c)
    rm = tl.load(rm_ptr + c)
    rv = tl.load(rv_ptr + c)

    # apply batchnorm using running stats (inference behavior) and produce pooled value
    denom = tl.sqrt(rv + eps)
    out_val = gamma * (mean_x - rm) / denom + beta

    # store result at out_ptr[n * C + c]
    tl.store(out_ptr + pid, out_val)


def triton_mean_bn_pool(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
                        running_mean: torch.Tensor, running_var: torch.Tensor,
                        scale: float, eps: float):
    """
    Compute fused: (x * scale) -> BatchNorm (inference using running stats) -> GlobalAvgPool
    Returns a tensor of shape (N, C, 1, 1, 1)

    Implementation uses a single Triton kernel that launches one program per (n, c).
    """
    assert x.is_cuda and gamma.is_cuda and beta.is_cuda and running_mean.is_cuda and running_var.is_cuda, \
        "All tensors must be on CUDA"
    # x expected shape: (N, C, D, H, W)
    x = x.contiguous()
    N, C, D, H, W = x.shape
    L = D * H * W

    # flatten spatial dims to shape (N, C, L) contiguous layout
    x_flat = x.view(N, C, L).contiguous()

    # prepare output tensor (N*C) -> then reshape to (N, C, 1, 1, 1)
    out = torch.empty((N * C,), device=x.device, dtype=x.dtype)

    # ensure parameter tensors are contiguous 1D tensors on CUDA
    gamma_c = gamma.contiguous()
    beta_c = beta.contiguous()
    rm_c = running_mean.contiguous()
    rv_c = running_var.contiguous()

    # grid is one program per (n,c) pair: launch exactly N*C program instances
    grid = lambda meta: (N * C,)

    # Launch fused Triton kernel; pass BLOCK as a constexpr parameter.
    _mean_bn_pool_kernel[grid](
        x_flat, out,
        gamma_c, beta_c, rm_c, rv_c,
        scale, eps,
        N, C, L,
        BLOCK=FUSED_BLOCK
    )

    # reshape to (N, C, 1, 1, 1)
    out = out.view(N, C, 1, 1, 1)
    return out


class ModelNew(nn.Module):
    """
    Optimized model:
      - uses the original ConvTranspose3d
      - fuses scale + BatchNorm (inference) + GlobalAvgPool into a single Triton kernel
      - falls back to PyTorch ops when batchnorm is in training mode
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        x = self.conv_transpose(x)
        # If batch_norm is in training mode, use PyTorch's original ops (so running stats and gradients are correct).
        if self.batch_norm.training:
            x = x * self.scale_factor
            x = self.batch_norm(x)
            x = self.global_avg_pool(x)
            return x
        else:
            # In inference mode, use fused Triton kernel:
            # The kernel computes mean over spatial dims of (x * scale), applies batchnorm using running stats,
            # and returns (N, C, 1, 1, 1).
            out = triton_mean_bn_pool(
                x,
                self.batch_norm.weight,
                self.batch_norm.bias,
                self.batch_norm.running_mean,
                self.batch_norm.running_var,
                float(self.scale_factor),
                float(self.batch_norm.eps),
            )
            return out


# keep helper functions for compatibility with the original interface
batch_size = 16
in_channels = 64
out_channels = 128
depth, height, width = 16, 32, 32
kernel_size = 5
scale_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]