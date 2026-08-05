import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configurations for different block sizes / warps
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 256},  num_warps=2, num_stages=2),
    triton.Config({"BLOCK": 512},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=3),
]

@triton.autotune(
    configs=AUTOTUNE_CONFIGS,
    key=["n_elements"],
)
@triton.jit
def mish_double_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    """
    Applies Mish activation twice elementwise: y = Mish(Mish(x))
    Operates on flattened 1D memory region of length n_elements.
    BLOCK is a compile-time block size.
    """
    pid = tl.program_id(0)
    start = pid * BLOCK
    offs = start + tl.arange(0, BLOCK)
    mask = offs < n_elements

    # Load input (use other=0.0 for masked lanes)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = x

    # Apply Mish twice
    for _ in range(2):
        # softplus: stable branch using tl.log and tl.exp (avoid log1p)
        pos_mask = y > 0.0
        exp_neg = tl.exp(-y)
        exp_pos = tl.exp(y)
        sp_pos = y + tl.log(1.0 + exp_neg)   # for y > 0
        sp_neg = tl.log(1.0 + exp_pos)       # for y <= 0
        sp = tl.where(pos_mask, sp_pos, sp_neg)

        # tanh(sp) computed via sigmoid trick: tanh(z) = 2*sigmoid(2z) - 1
        denom = 1.0 + tl.exp(-2.0 * sp)
        tanh_sp = 2.0 / denom - 1.0

        # Mish: x * tanh(softplus(x))
        y = y * tanh_sp

    # Store output
    tl.store(out_ptr + offs, y, mask=mask)

def triton_mish_double(x: torch.Tensor):
    """
    Wrapper to apply the Triton kernel to tensor x (any shape).
    Returns a new tensor with Mish applied twice elementwise.
    """
    assert x.is_cuda, "Input must be on CUDA."
    x_contig = x.contiguous()
    out = torch.empty_like(x_contig)
    n_elements = x_contig.numel()

    # grid: number of program instances based on block selected by autotuner
    grid = lambda meta: ( (n_elements + meta["BLOCK"] - 1) // meta["BLOCK"], )

    mish_double_kernel[grid](x_contig, out, n_elements)
    return out

class ModelNew(nn.Module):
    """
    Optimized model: uses PyTorch Conv2d for convolution and a fused Triton kernel
    to apply Mish activation twice in a single elementwise GPU kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        # Fuse the two Mish calls into one Triton kernel pass
        x = triton_mish_double(x)
        return x

# Keep helper functions to match expected interface for input generation
batch_size   = 64
in_channels  = 64
out_channels = 128
height = width = 256
kernel_size = 3

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]