import torch
import torch.nn as nn
import triton
import triton.language as tl

# Triton kernel that applies per-channel BatchNorm (using running_mean/running_var and affine params)
# and then multiplies by a global scaling factor. The kernel processes BLOCK spatial elements at a time.
@triton.jit
def _bn_scale_kernel(
    x_ptr,            # pointer to input tensor (N*C*H*W)
    out_ptr,          # pointer to output tensor (N*C*H*W)
    gamma_ptr,        # pointer to bn weight (C,)
    beta_ptr,         # pointer to bn bias (C,)
    mean_ptr,         # pointer to running_mean (C,)
    var_ptr,          # pointer to running_var (C,)
    N,                # batch size
    C,                # number of channels
    H,                # height
    W,                # width
    eps,              # epsilon for batchnorm (float32)
    scale,            # global scaling factor (float32)
    BLOCK: tl.constexpr
):
    # program ids
    n_idx = tl.program_id(0)           # 0..N-1
    c_idx = tl.program_id(1)           # 0..C-1
    block_spatial = tl.program_id(2)   # spatial block id

    spatial_size = H * W
    block_start = block_spatial * BLOCK
    offs = block_start + tl.arange(0, BLOCK)  # vector of size BLOCK
    mask = offs < spatial_size

    # compute base pointer offset for (n, c, 0)
    base = (n_idx * C + c_idx) * spatial_size
    idxs = base + offs  # these are flattened element indices into N*C*H*W layout

    # load input values (with mask)
    x = tl.load(x_ptr + idxs, mask=mask, other=0.0)

    # load per-channel params (scalars)
    mean = tl.load(mean_ptr + c_idx)
    var = tl.load(var_ptr + c_idx)
    gamma = tl.load(gamma_ptr + c_idx)
    beta = tl.load(beta_ptr + c_idx)

    inv = 1.0 / tl.sqrt(var + eps)

    # apply batchnorm: (x - mean) * inv * gamma + beta, then scale
    y = (x - mean) * inv * gamma + beta
    y = y * scale

    # store results
    tl.store(out_ptr + idxs, y, mask=mask)


def _bn_scale_triton(x: torch.Tensor,
                     gamma: torch.Tensor,
                     beta: torch.Tensor,
                     running_mean: torch.Tensor,
                     running_var: torch.Tensor,
                     eps: float,
                     scale: float):
    """
    Wrapper to launch the Triton kernel that applies BatchNorm (using running stats)
    and a final global scaling factor. Expects x on CUDA, contiguous, dtype float32.
    """
    assert x.is_cuda, "Input must be on CUDA"
    assert x.dtype == torch.float32, "Only float32 is supported"

    N, C, H, W = x.shape
    spatial = H * W

    # Ensure parameter tensors are on the same device and contiguous
    device = x.device
    if gamma is None:
        gamma = torch.ones(C, device=device, dtype=torch.float32)
    if beta is None:
        beta = torch.zeros(C, device=device, dtype=torch.float32)

    gamma = gamma.contiguous().to(device=device, dtype=torch.float32)
    beta = beta.contiguous().to(device=device, dtype=torch.float32)
    running_mean = running_mean.contiguous().to(device=device, dtype=torch.float32)
    running_var = running_var.contiguous().to(device=device, dtype=torch.float32)

    x_cont = x.contiguous()
    out = torch.empty_like(x_cont)

    # Choose a BLOCK size tuned for spatial locality; 256 is a reasonable default.
    BLOCK = 256
    num_blocks = (spatial + BLOCK - 1) // BLOCK

    grid = (N, C, num_blocks)

    # Launch the Triton kernel. Provide BLOCK as a constexpr.
    _bn_scale_kernel[grid](
        x_cont, out,
        gamma, beta, running_mean, running_var,
        N, C, H, W,
        float(eps), float(scale),
        BLOCK=BLOCK
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized model that keeps the PyTorch convolution (leveraging its high-performance
    implementation) but replaces the subsequent BatchNorm + scaling with a fused
    Triton kernel for improved throughput on GPU.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(ModelNew, self).__init__()
        # Keep the conv as a standard PyTorch Conv2d for correctness and performance.
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # Keep BatchNorm object to hold parameters/buffers (we will use them in the Triton kernel).
        self.bn = nn.BatchNorm2d(out_channels)
        # scaling factor (float)
        self.scaling_factor = float(scaling_factor)

    def forward(self, x: torch.Tensor):
        # Perform convolution with PyTorch implementation (highly optimized).
        x = self.conv(x)
        # Determine whether to use batch statistics (training) or running statistics (eval).
        # If in training mode, compute per-channel mean/var from the convolution output
        # so that the Triton kernel matches PyTorch BatchNorm behavior.
        if self.bn.training:
            # x shape: (N, C, H, W). Compute mean/var over (N, H, W) for each channel.
            mean = x.mean(dim=(0, 2, 3)).contiguous()
            var = x.var(dim=(0, 2, 3), unbiased=False).contiguous()
        else:
            mean = self.bn.running_mean
            var = self.bn.running_var
        # Use Triton kernel to apply BatchNorm (using chosen stats and affine params) and scaling.
        return _bn_scale_triton(
            x,
            self.bn.weight if self.bn.affine else None,
            self.bn.bias if self.bn.affine else None,
            mean,
            var,
            float(self.bn.eps),
            float(self.scaling_factor)
        )