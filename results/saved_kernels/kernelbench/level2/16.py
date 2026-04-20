import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configurations for the Triton kernel
# Keep balanced BLOCK_SIZE candidates (multiples of warp size) and reasonable
# warp counts/stages for Ampere (A6000).
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=8, num_stages=2),
]

@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['n_elements'])
@triton.jit
def _mish_add_hardtanh_scale_kernel(
    x_ptr,          # pointer to input (fp32)
    out_ptr,        # pointer to output (fp32)
    n_elements,     # total number of elements
    add_value,      # scalar to add (fp32)
    min_val,        # hardtanh min (fp32)
    max_val,        # hardtanh max (fp32)
    scale,          # scalar to scale (fp32)
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the block this program will handle
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input as FP32
    x_fp32 = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Piecewise handling to avoid unnecessary exp/log work:
    # For large positive x (> +20): softplus(x) ≈ x => tanh(softplus) ≈ 1 => mish ≈ x
    # For large negative x (< -20): softplus(x) ≈ exp(x) (very small) => mish ≈ x * exp(x) (tiny)
    # Compute transcendental functions in FP32 to avoid invalid FP16 transcendental calls.
    large_mask = x_fp32 > 20.0
    small_mask = x_fp32 < -20.0

    # Compute mid-range path in FP32
    exp_x = tl.exp(x_fp32)
    sp_mid = tl.log(1.0 + exp_x)           # softplus in FP32
    neg2sp = -2.0 * sp_mid
    exp_neg2sp = tl.exp(neg2sp)
    tanh_sp = (1.0 - exp_neg2sp) / (1.0 + exp_neg2sp)
    mish_mid = x_fp32 * tanh_sp

    # Large positive approximation: mish ≈ x
    mish_large = x_fp32

    # Large negative approximation: softplus ≈ exp(x), mish ≈ x * exp(x)
    mish_small = x_fp32 * exp_x

    # Select piecewise result in FP32
    mish_fp32 = tl.where(large_mask, mish_large, tl.where(small_mask, mish_small, mish_mid))

    # Add the scalar value
    val = mish_fp32 + add_value

    # Hardtanh clamp (FP32)
    val = tl.maximum(val, min_val)
    val = tl.minimum(val, max_val)

    # Scale
    val = val * scale

    # Store result
    tl.store(out_ptr + offsets, val, mask=mask)

def triton_mish_add_hardtanh_scale(x: torch.Tensor, add_value: float, min_val: float, max_val: float, scale: float):
    """
    Apply fused Mish -> add -> Hardtanh -> scale using Triton kernel.
    Falls back to PyTorch if tensor is not on CUDA or not float32.
    """
    if not x.is_cuda or x.dtype != torch.float32:
        # Fallback to PyTorch implementation on CPU or non-fp32 tensors
        y = torch.nn.functional.mish(x)
        y = y + add_value
        y = torch.nn.functional.hardtanh(y, min_val=min_val, max_val=max_val)
        y = y * scale
        return y

    # Ensure contiguous
    x_contig = x.contiguous()
    out = torch.empty_like(x_contig)

    n_elements = x_contig.numel()
    if n_elements == 0:
        return out

    # Grid based on BLOCK_SIZE chosen by autotune
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch Triton kernel
    _mish_add_hardtanh_scale_kernel[grid](
        x_contig, out, n_elements,
        float(add_value), float(min_val), float(max_val), float(scale)
    )
    return out

class ModelNew(nn.Module):
    """
    Optimized model: keep ConvTranspose2d (Cuda-optimized in PyTorch), fuse Mish + add + Hardtanh + scale into a single Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        # store the scalar parameters
        self.add_value = float(add_value)
        self.scale = float(scale)
        # hardtanh bounds are fixed as in the original model
        self.hardtanh_min = -1.0
        self.hardtanh_max = 1.0

    def forward(self, x):
        x = self.conv_transpose(x)
        # Fuse Mish + add + Hardtanh + scale with Triton kernel when possible
        y = triton_mish_add_hardtanh_scale(x, self.add_value, self.hardtanh_min, self.hardtanh_max, self.scale)
        return y