import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configurations for the Triton kernel
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_P": 64},  num_warps=2, num_stages=2),
    triton.Config({"BLOCK_P": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_P": 256}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_P": 512}, num_warps=8, num_stages=3),
]

@triton.autotune(
    configs=AUTOTUNE_CONFIGS,
    key=["N", "C", "H", "W", "pooled_H", "pooled_W", "pool_k"],
)
@triton.jit
def fused_pool_tanh_scale_bias_kernel(
    x_ptr,          # input tensor pointer (N*C*H*W elements)
    bias_ptr,       # bias per channel pointer (C elements)
    out_ptr,        # output pointer (N*C*pooled_H*pooled_W elements)
    N, C, H, W,     # input dimensions
    pooled_H, pooled_W,  # output pooled dimensions
    scaling,        # scaling factor (float)
    total_out,      # total number of output elements (N*C*pooled_H*pooled_W)
    pool_k: tl.constexpr,  # pooling kernel size (constexpr int)
    BLOCK_P: tl.constexpr,
):
    """
    Each program computes up to BLOCK_P output pooled elements.
    For each output (n, c, ph, pw) we compute:
      max_{i=0..pool_k-1, j=0..pool_k-1} (tanh(x[n,c,ph*pool_k+i,pw*pool_k+j]) * scaling + bias[c])
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_P
    offs = block_start + tl.arange(0, BLOCK_P)
    mask = offs < total_out  # which outputs in this block are valid

    # Precompute some sizes
    elems_per_image = C * pooled_H * pooled_W  # number of pooled outputs per batch element
    pooled_HW = pooled_H * pooled_W

    # Map linear output index offs -> (n_idx, c_idx, ph, pw)
    n_idx = offs // elems_per_image
    rem = offs - n_idx * elems_per_image
    c_idx = rem // pooled_HW
    rem2 = rem - c_idx * pooled_HW
    ph = rem2 // pooled_W
    pw = rem2 - ph * pooled_W

    # Compute base input coordinates
    base_i = ph * pool_k
    base_j = pw * pool_k

    # Load bias per channel for each element (masked)
    bias_vals = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

    # Initialize accumulator with very small value so max works correctly
    neg_inf = -1e20
    acc = tl.full((BLOCK_P,), neg_inf, dtype=tl.float32)

    # For each position in the pooling window, load, apply tanh, scale, add bias, and update max
    # Unroll exactly pool_k x pool_k iterations by making pool_k a constexpr
    for i in range(pool_k):
        for j in range(pool_k):
            # compute current coordinates
            cur_i = base_i + i
            cur_j = base_j + j

            # Compose input linear index: ((n * C + c) * H + cur_i) * W + cur_j
            idx1 = n_idx * C + c_idx  # (BLOCK_P,)
            idx2 = idx1 * H + cur_i
            pos = idx2 * W + cur_j  # linear index into input flattened N*C*H*W

            # mask for valid loads: must be output valid and within bounds
            load_mask = mask & (cur_i < H) & (cur_j < W)

            # Load x values (use a very negative other to not affect max for invalid lanes)
            x_vals = tl.load(x_ptr + pos, mask=load_mask, other=-1e19)

            # Compute tanh(x) using single-exp, sign-based formula:
            # tanh(x) = sign(x) * (e - 1) / (e + 1), where e = exp(2 * abs(x))
            e = tl.exp(2.0 * tl.abs(x_vals))
            sign = tl.where(x_vals >= 0.0, 1.0, -1.0)
            tanh_vals = sign * (e - 1.0) / (e + 1.0)

            # Apply scaling and bias
            vals = tanh_vals * scaling + bias_vals

            # Update accumulator with element-wise maximum
            acc = tl.where(vals > acc, vals, acc)

    # Store the result into output (flat indexing)
    tl.store(out_ptr + offs, acc, mask=mask)


def triton_fused_pool_tanh_scale_bias(x: torch.Tensor, bias: torch.Tensor, scaling: float, pool_k: int):
    """
    Wrapper to call the Triton kernel that fuses tanh, scaling, bias addition, and max-pooling.
    x: tensor of shape (N, C, H, W), float32, CUDA
    bias: tensor of shape (C,) or (C,1,1)
    scaling: float scaling factor
    pool_k: int pooling kernel size
    """
    assert x.is_cuda and bias.is_cuda, "Inputs must be on CUDA."
    assert x.dtype == torch.float32 and bias.dtype == torch.float32, "Only float32 supported."

    x_contig = x.contiguous()
    N, C, H, W = x_contig.shape

    # Compute pooled output size consistent with PyTorch MaxPool2d(kernel_size=pool_k, stride=pool_k)
    pooled_H = (H - pool_k) // pool_k + 1
    pooled_W = (W - pool_k) // pool_k + 1

    # Flatten bias to shape (C,)
    if bias.ndim == 3:
        bias_flat = bias.view(-1).contiguous()
    else:
        bias_flat = bias.contiguous()

    # Prepare output tensor
    out = torch.empty((N, C, pooled_H, pooled_W), device=x.device, dtype=x.dtype)

    total_out = N * C * pooled_H * pooled_W

    # grid based on BLOCK_P
    grid = lambda meta: ((total_out + meta["BLOCK_P"] - 1) // meta["BLOCK_P"],)

    # Launch Triton kernel
    fused_pool_tanh_scale_bias_kernel[grid](
        x_contig, bias_flat, out, N, C, H, W, pooled_H, pooled_W, float(scaling), total_out, pool_k
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized model that uses PyTorch Conv2d for convolution and a Triton kernel to fuse
    tanh, scaling, bias addition, and max-pooling into a single pass over the convolution output.
    This reduces memory traffic and kernel launches compared to separate PyTorch ops.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = scaling_factor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.pool_kernel = pool_kernel_size  # integer

    def forward(self, x):
        # Convolution using PyTorch (efficient)
        x = self.conv(x)
        # Fused tanh, scaling, bias addition, and max-pooling using Triton
        bias_flat = self.bias
        if bias_flat.device != x.device:
            bias_flat = bias_flat.to(x.device)
        bias_flat = bias_flat.view(-1)
        x = triton_fused_pool_tanh_scale_bias(x, bias_flat, self.scaling_factor, self.pool_kernel)
        return x