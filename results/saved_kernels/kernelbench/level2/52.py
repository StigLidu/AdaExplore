import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configs for the elementwise Triton kernel
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2),
]

@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['n_elements'])
@triton.jit
def _softplus_tanh_mul_kernel(
    x_ptr,            # pointer to input
    out_ptr,          # pointer to output
    n_elements,       # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load input with mask and a safe other value
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # numerically stable softplus: softplus(x) = max(0,x) + log(1 + exp(-|x|))
    pos = tl.where(x > 0.0, x, 0.0)
    neg_abs = -tl.abs(x)
    # compute log(1 + exp(neg_abs)) as tl.log(1.0 + tl.exp(neg_abs))
    sp = pos + tl.log(1.0 + tl.exp(neg_abs))

    # tanh(sp) computed via exp to avoid using tl.tanh
    e2 = tl.exp(2.0 * sp)
    tanh_sp = (e2 - 1.0) / (e2 + 1.0)

    out = x * tanh_sp

    tl.store(out_ptr + offs, out, mask=mask)

def triton_softplus_tanh_mul(x: torch.Tensor) -> torch.Tensor:
    """
    Applies elementwise: x * tanh(softplus(x)) using a Triton kernel.
    """
    assert x.is_cuda, "Input must be a CUDA tensor."
    # Ensure contiguous for efficient pointer access
    x_contig = x.contiguous()
    out = torch.empty_like(x_contig)

    n_elements = x_contig.numel()
    # grid based on BLOCK_SIZE chosen by autotuner
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch kernel on flattened views
    _softplus_tanh_mul_kernel[grid](
        x_contig.view(-1),
        out.view(-1),
        n_elements,
    )
    return out

class ModelNew(nn.Module):
    """
    Optimized model that uses a Triton kernel to compute the activation:
      f(x) = x * tanh(softplus(x))
    The convolution and batchnorm use PyTorch implementations to keep correctness
    for training/eval semantics of BatchNorm2d.
    """
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        # Keep same conv and batchnorm definitions to preserve behavior.
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)

    def forward(self, x):
        # Convolution using PyTorch's optimized kernel
        x = self.conv(x)
        # Fused elementwise activation via Triton
        x = triton_softplus_tanh_mul(x)
        # BatchNorm (keeps training/eval semantics)
        x = self.bn(x)
        return x