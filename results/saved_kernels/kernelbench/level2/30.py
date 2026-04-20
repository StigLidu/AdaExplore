import torch
import torch.nn as nn
import triton
import triton.language as tl

# Triton kernel that fuses Group Normalization (per-sample, per-group) and HardTanh (clamp)
# Assumptions:
# - Input is a contiguous FP32 tensor of shape (N, C).
# - C is divisible by num_groups.
# - BLOCK (channels per group) is passed as a constexpr so we can call tl.arange(BLOCK).
@triton.jit
def _groupnorm_hardtanh_kernel(
    x_ptr,            # input pointer (N, C) row-major
    out_ptr,          # output pointer (N, C) row-major
    gamma_ptr,        # groupnorm weight (C,)
    beta_ptr,         # groupnorm bias (C,)
    N,                # batch size
    C,                # num channels
    G,                # num groups
    eps,              # epsilon for numerical stability
    lower,            # hardtanh min
    upper,            # hardtanh max
    BLOCK: tl.constexpr,  # channels per group (C // G), constexpr
):
    pid = tl.program_id(0)                        # one program per (sample, group)
    n = pid // G                                  # sample index
    g = pid % G                                   # group index

    c_start = g * BLOCK
    offs = c_start + tl.arange(0, BLOCK)          # offsets within the channel dimension
    row_off = n * C                               # offset to the start of the sample

    # pointers to the BLOCK elements for this (n, g)
    ptrs = x_ptr + row_off + offs
    vals = tl.load(ptrs)

    # compute mean
    mean = tl.sum(vals) / BLOCK

    # center and variance
    centered = vals - mean
    var = tl.sum(centered * centered) / BLOCK

    invstd = 1.0 / tl.sqrt(var + eps)

    # load affine parameters for these channels
    gamma = tl.load(gamma_ptr + offs)
    beta = tl.load(beta_ptr + offs)

    # apply normalization, affine transform, and clamp (hardtanh)
    out_vals = (centered * invstd) * gamma + beta
    out_vals = tl.where(out_vals < lower, lower, out_vals)
    out_vals = tl.where(out_vals > upper, upper, out_vals)

    tl.store(out_ptr + row_off + offs, out_vals)


def triton_groupnorm_hardtanh(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, num_groups: int, lower: float, upper: float, eps: float = 1e-5):
    """
    Fuses GroupNorm (per-sample, across channels grouped by `num_groups`) and HardTanh.
    x: (N, C) contiguous FP32 tensor on CUDA
    gamma, beta: (C,) tensors (affine parameters)
    Returns: (N, C) contiguous FP32 tensor
    """
    assert x.is_cuda and gamma.is_cuda and beta.is_cuda, "All tensors must be on CUDA."
    assert x.dtype == torch.float32 and gamma.dtype == torch.float32 and beta.dtype == torch.float32, "Only FP32 supported."
    N, C = x.shape
    assert C % num_groups == 0, "num_groups must divide number of channels"
    c_per_group = C // num_groups

    # Ensure contiguity for pointer arithmetic in Triton kernel
    x = x.contiguous()
    gamma = gamma.contiguous()
    beta = beta.contiguous()

    out = torch.empty_like(x)
    # grid: one program per (sample, group)
    grid = (N * num_groups,)

    # Launch the Triton kernel; pass BLOCK as constexpr
    _groupnorm_hardtanh_kernel[grid](
        x, out, gamma, beta,
        N, C, num_groups,
        float(eps),
        float(lower),
        float(upper),
        BLOCK=c_per_group
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that keeps PyTorch's fast linear (GEMM) and replaces the
    GroupNorm + HardTanh sequence with a fused Triton kernel for improved memory locality.
    Behavior is equivalent to the original Model:
      - Linear (nn.Linear)
      - GroupNorm (nn.GroupNorm) with affine parameters
      - HardTanh (clamp to [min, max])
    """
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        # Keep the Linear module to register weight and bias and use cuBLAS GEMM
        self.gemm = nn.Linear(in_features, out_features)
        # Keep GroupNorm module to register affine parameters (weight and bias) and num_groups
        # We will not call its forward; instead we will use its parameters in a custom Triton kernel.
        self.group_norm = nn.GroupNorm(num_groups, out_features, eps=1e-5, affine=True)
        self.hardtanh_min = float(hardtanh_min)
        self.hardtanh_max = float(hardtanh_max)

        # Ensure that the GroupNorm's affine params are initialized in the default PyTorch way
        # (nn.GroupNorm does this in its constructor). We do not change them.

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch_size, in_features) FP32 CUDA tensor
        Returns:
            (batch_size, out_features) tensor
        """
        # Use PyTorch's highly-optimized linear (GEMM) for the heavy matrix multiplication
        y = self.gemm(x)  # shape (N, C)

        # Fuse GroupNorm (per-sample, per-group) and HardTanh via Triton kernel
        gamma = self.group_norm.weight  # shape (C,)
        beta = self.group_norm.bias     # shape (C,)
        out = triton_groupnorm_hardtanh(y, gamma, beta, self.group_norm.num_groups, self.hardtanh_min, self.hardtanh_max, eps=self.group_norm.eps)
        return out