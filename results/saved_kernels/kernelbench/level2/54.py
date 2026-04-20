import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotuning configs exploring a wide range of spatial block sizes and warps.
# Larger BLOCK_M reduces kernel launch overhead and increases vector throughput for large spatial planes.
# Prefer BLOCK_M values that are multiples of 32/4 for better vectorization and coalescing on Ampere GPUs.
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 128}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 256}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 512}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 1024}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 2048}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['N', 'C', 'HW'])
@triton.jit
def fused_mul_leaky_gelu_2d(
    x_ptr,        # pointer to input (flattened)
    out_ptr,      # pointer to output (flattened)
    mult_ptr,     # pointer to multiplier per-channel (length C)
    N,            # batch size
    C,            # channels (out_channels)
    HW,           # spatial size H*W
    negative_slope,  # leaky relu negative slope (float)
    BLOCK_M: tl.constexpr,
):
    """
    Each program handles one (n,c) pair and a contiguous BLOCK_M slice of the spatial plane.
    Grid dims:
      0 -> N    (batch dimension)
      1 -> C    (channel dimension)
      2 -> ceil(HW / BLOCK_M) (partition spatial plane)
    """
    n_id = tl.program_id(0)      # batch index 0..N-1
    c_id = tl.program_id(1)      # channel index 0..C-1
    m_block = tl.program_id(2)   # which spatial block (0 .. (HW-1)//BLOCK_M)

    # spatial offsets within the H*W plane for this block
    offs_in_block = tl.arange(0, BLOCK_M)
    spatial_offset = m_block * BLOCK_M + offs_in_block  # length BLOCK_M
    mask = spatial_offset < HW

    # base pointer for this (n,c) pair: (n_id * C + c_id) * HW
    base = (n_id * C + c_id) * HW
    abs_offs = base + spatial_offset

    # Load input block (masked). other must be provided.
    vals = tl.load(x_ptr + abs_offs, mask=mask, other=0.0)

    # load multiplier once for this channel
    mult = tl.load(mult_ptr + c_id)

    # apply multiplier (broadcast)
    vals = vals * mult

    # LeakyReLU
    is_neg = vals < 0.0
    vals = tl.where(is_neg, vals * negative_slope, vals)

    # GELU approximation using Triton's sigmoid to reduce math cost:
    # x * sigmoid(1.702 * x)
    y = 1.702 * vals
    sig = tl.sigmoid(y)
    out = vals * sig

    # Store result
    tl.store(out_ptr + abs_offs, out, mask=mask)


def fused_mul_leaky_gelu(x: torch.Tensor, multiplier: torch.Tensor, negative_slope: float = 0.01):
    """
    Wrapper for the Triton fused kernel.
    x: tensor of shape [N, C, H, W], contiguous on CUDA
    multiplier: tensor of shape [C] or [C,1,1], device matching x
    """
    assert x.is_cuda, "Input must be on CUDA"
    assert multiplier.is_cuda, "Multiplier must be on CUDA"

    x_contig = x.contiguous()
    N, C, H, W = x_contig.shape
    HW = H * W
    # Grid will be 3D: (N, C, num_spatial_blocks)
    n_programs_nc = N * C

    # Flatten tensors for pointer arithmetic
    x_flat = x_contig.view(-1)
    out = torch.empty_like(x_contig)
    out_flat = out.view(-1)

    # Prepare multiplier as 1D contiguous vector length C on correct device/dtype
    mult_flat = multiplier.contiguous().view(-1).to(x_contig.device).to(x_contig.dtype)

    # autotuned grid: programs (N, C, ceil(HW/BLOCK_M))
    grid = lambda meta: (N, C, (HW + meta["BLOCK_M"] - 1) // meta["BLOCK_M"])

    fused_mul_leaky_gelu_2d[grid](
        x_flat, out_flat, mult_flat, N, C, HW, float(negative_slope)
    )

    return out


class ModelNew(nn.Module):
    """
    Model optimized with a fused Triton kernel that performs:
      (conv output) * multiplier -> LeakyReLU -> GELU (approx)
    The convolution itself uses PyTorch's optimized nn.Conv2d.
    """

    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # Keep multiplier in original requested shape, but we'll pass a flattened view to Triton
        # multiplier_shape expected like (out_channels, 1, 1)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape, dtype=torch.float32))

    def forward(self, x):
        # Run convolution (use PyTorch's implementation)
        x = self.conv(x)

        # Prepare multiplier as per-channel vector on same device/dtype as x
        mult = self.multiplier.view(self.multiplier.shape[0]).to(x.device).to(x.dtype)

        # Run fused Triton kernel for multiply + LeakyReLU + GELU (approx)
        x = fused_mul_leaky_gelu(x, mult, negative_slope=0.01)
        return x