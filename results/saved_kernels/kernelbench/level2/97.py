import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotuning configurations for the fused elementwise kernel
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 512},  num_warps=4, num_stages=2),
]


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["n_elements"])
@triton.jit
def _fused_bias_divide_swish_kernel(
    inp_ptr,        # pointer to input (flattened)
    out_ptr,        # pointer to output (flattened)
    bias_ptr,       # pointer to bias scalar (shape [1])
    inv_div,        # scalar float32: reciprocal of divide_value (1.0 / divide_value)
    n_elements,     # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    """
    For each element:
      v = inp[i]
      v = v + bias
      v = v * inv_div   # use multiply by reciprocal instead of division
      out[i] = v * sigmoid(v)
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load inputs with masking
    x = tl.load(inp_ptr + offsets, mask=mask, other=0.0)

    # Load bias scalar (broadcasted)
    bias = tl.load(bias_ptr)

    # Apply operations
    x = x + bias
    x = x * inv_div
    # sigmoid: 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-x))
    out = x * sig

    # Store results
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_fused_bias_divide_swish(x: torch.Tensor, bias: torch.Tensor, divide_value: float):
    """
    Wrapper that launches the Triton kernel to perform:
      out = ((x + bias) / divide_value) * sigmoid((x + bias) / divide_value)
    x: Tensor of shape (N, C) or arbitrary shape; will be flattened for kernel.
    bias: Tensor of shape (1,) (broadcast scalar).
    divide_value: Python float or scalar convertible to float.
    """
    if not x.is_cuda or not bias.is_cuda:
        # Fallback to PyTorch if not on CUDA
        y = x + bias
        y = y / divide_value
        return y * torch.sigmoid(y)

    # Avoid unnecessary copy if x is already contiguous
    if x.is_contiguous():
        x_flat = x.view(-1)
    else:
        x_flat = x.contiguous().view(-1)

    # Prepare output with the same shape/device/dtype as x
    out = torch.empty_like(x)
    out_flat = out.view(-1)
    bias_contig = bias.contiguous()

    n_elements = x_flat.numel()

    # Precompute reciprocal on host to avoid per-element division in the kernel
    inv_div = 1.0 / float(divide_value)

    # Grid based on BLOCK_SIZE chosen by autotune
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    _fused_bias_divide_swish_kernel[grid](
        x_flat, out_flat, bias_contig, inv_div, n_elements
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized model that reuses PyTorch's linear and BatchNorm1d but fuses
    the final bias addition, division, and Swish activation into a single
    Triton kernel for efficient elementwise processing on the GPU.
    """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        super(ModelNew, self).__init__()
        # Keep Linear and BatchNorm1d to leverage their optimized implementations
        self.matmul = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        # Additional bias (broadcast scalar by default)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.divide_value = float(divide_value)

    def forward(self, x):
        # x @ W^T + b_linear
        x = self.matmul(x)
        # BatchNorm (uses running stats in eval, computes batch stats in train)
        x = self.bn(x)
        # Fuse bias add, division, and Swish into a single Triton kernel if on CUDA
        if x.is_cuda and self.bias.is_cuda:
            return triton_fused_bias_divide_swish(x, self.bias, self.divide_value)
        else:
            # CPU or mixed device fallback: keep behavior identical
            y = x + self.bias
            y = y / self.divide_value
            return y * torch.sigmoid(y)