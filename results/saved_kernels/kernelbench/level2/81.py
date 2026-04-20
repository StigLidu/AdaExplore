import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Autotune configurations for the fused elementwise kernel
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 128},  num_warps=2, num_stages=2),
    triton.Config({"BLOCK_SIZE": 256},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 512},  num_warps=8, num_stages=2),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['n_elements'])
@triton.jit
def _fused_swish_div_clamp_tanh_kernel(
    inp_ptr,         # input pointer (y = linear(x))
    out_ptr,         # output pointer
    n_elements,      # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load input
    x = tl.load(inp_ptr + offs, mask=mask, other=0.0)

    # Swish: x * sigmoid(x) where sigmoid(x) = 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig

    # Divide by 2
    y = y * 0.5

    # Clamp between -1 and 1
    y = tl.where(y < -1.0, -1.0, y)
    y = tl.where(y > 1.0, 1.0, y)

    # Tanh via stable formulation: tanh(z) = (1 - exp(-2z)) / (1 + exp(-2z))
    e = tl.exp(-2.0 * y)
    t = (1.0 - e) / (1.0 + e)

    # Final clamp between -1 and 1 (redundant numerically but kept to match original)
    t = tl.where(t < -1.0, -1.0, t)
    t = tl.where(t > 1.0, 1.0, t)

    # Store result
    tl.store(out_ptr + offs, t, mask=mask)


def fused_swish_div_clamp_tanh(x: torch.Tensor):
    """
    Wrapper that launches the Triton kernel to compute:
        out = clamp(tanh(clamp((x * sigmoid(x)) / 2, -1, 1)), -1, 1)
    on all elements of x.
    """
    assert x.is_cuda, "Input must be a CUDA tensor."
    x = x.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    # grid depends on selected BLOCK_SIZE by autotune
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    # Launch kernel (autotune will pick BLOCK_SIZE)
    _fused_swish_div_clamp_tanh_kernel[grid](x, out, n_elements)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that uses Triton to fuse the elementwise operations
    (swish, divide, clamp, tanh, clamp) into a single GPU kernel.
    The linear (GEMM) is still executed by torch's highly optimized implementation.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        # keep a standard Linear layer for the GEMM (uses cuBLAS/cuDNN)
        self.gemm = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        # Compute GEMM using PyTorch (fast cuBLAS)
        y = F.linear(x, self.gemm.weight, self.gemm.bias)

        # Apply fused elementwise operations via Triton kernel
        y = fused_swish_div_clamp_tanh(y)
        return y


# Helper functions to match the original style
batch_size = 1024
in_features = 8192
out_features = 8192

def get_inputs():
    # Return a CUDA tensor since the Triton kernel expects CUDA inputs
    return [torch.rand(batch_size, in_features, device='cuda', dtype=torch.float32)]

def get_init_inputs():
    return [in_features, out_features]