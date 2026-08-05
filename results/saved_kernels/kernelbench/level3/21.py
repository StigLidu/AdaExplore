import torch
import torch.nn as nn
import torch.nn.functional as F

# Triton imports for custom kernels
import triton
import triton.language as tl

# Autotune configs used by the Triton kernels
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 64},  num_warps=1, num_stages=2),
    triton.Config({"BLOCK_SIZE": 128}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=3),
]

# Triton kernel: elementwise clipped ReLU6 (y = min(max(x, 0), 6))
@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['n_elements'])
@triton.jit
def _relu6_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # relu6: clamp between 0 and 6
    zero = tl.zeros(x.shape, dtype=tl.float32)
    six = tl.full(x.shape, 6.0, dtype=tl.float32)
    x = tl.maximum(x, zero)
    x = tl.minimum(x, six)
    tl.store(out_ptr + offsets, x, mask=mask)

def triton_relu6(x: torch.Tensor) -> torch.Tensor:
    """
    Apply ReLU6 using a Triton kernel. Falls back to torch.relu6 if tensor is not CUDA.
    """
    if not x.is_cuda:
        return torch.clamp(x, 0.0, 6.0)

    xc = x.contiguous()
    out = torch.empty_like(xc)

    n_elements = xc.numel()
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    _relu6_kernel[grid](xc, out, n_elements)
    return out

# Triton kernel: elementwise addition (out = x + y)
@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['n_elements'])
@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)

def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Elementwise add using Triton. Falls back to torch.add if tensors are not CUDA.
    """
    if not (x.is_cuda and y.is_cuda):
        return x + y

    xa = x.contiguous()
    ya = y.contiguous()
    out = torch.empty_like(xa)

    n_elements = xa.numel()
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    _add_kernel[grid](xa, ya, out, n_elements)
    return out

class ModelNew(nn.Module):
    """
    MBConv-like block optimized to use Triton kernels for elementwise ReLU6 and residual add.
    The convolution and batchnorm layers remain as torch modules to leverage cuDNN/cuBLAS,
    while activation and the residual addition are handled by custom Triton kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super(ModelNew, self).__init__()

        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio

        # Expand (1x1 conv + BN) without activation; we'll apply Triton ReLU6 after BN
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
            )

        # Depthwise conv (KxK conv with groups=hidden_dim) + BN; activation applied via Triton
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride,
                      padding=(kernel_size - 1) // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
        )

        # Project (1x1 conv + BN)
        self.project_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        identity = x

        if hasattr(self, 'expand_conv'):
            x = self.expand_conv(x)
            # apply fused ReLU6 via Triton
            x = triton_relu6(x)

        x = self.depthwise_conv(x)
        x = triton_relu6(x)

        x = self.project_conv(x)

        if self.use_residual:
            # Use Triton for the elementwise residual addition
            x = triton_add(x, identity)

        return x