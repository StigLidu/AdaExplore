import torch
import torch.nn as nn
import triton
import triton.language as tl

# Hardware: A6000 - tune larger BLOCK sizes and warps/stages for Ampere
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 256},  num_warps=4, num_stages=3),
    triton.Config({"BLOCK": 512},  num_warps=4, num_stages=3),
    triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK": 4096}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK": 8192}, num_warps=8, num_stages=3),
]

@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['n_elements'])
@triton.jit
def _square_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BLOCK
    offs = start + tl.arange(0, BLOCK)
    mask = offs < n_elements
    vals = tl.load(x_ptr + offs, mask=mask, other=0.0)
    out = vals * vals
    tl.store(out_ptr + offs, out, mask=mask)

# Fused add+mul 2D tiled kernel: compute out = (x + y) * y with tiles (BLOCK_M x BLOCK_N)
# Autotune a selection of tile sizes that are multiples of 128/256 for features and small N tiles.
AUTOTUNE_2D_CONFIGS = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 8},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 8},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 16}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 16}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 32}, num_warps=8, num_stages=3),
]

@triton.autotune(configs=AUTOTUNE_2D_CONFIGS, key=['M', 'N'])
@triton.jit
def _fused_add_mul_kernel(x_ptr, y_ptr, out_ptr, M, N, stride_x, stride_y, stride_out, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # program id 0 -> over N (rows / batch), program id 1 -> over M (columns / features)
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    row_start = pid_n * BLOCK_N
    col_start = pid_m * BLOCK_M

    offs_n = row_start + tl.arange(0, BLOCK_N)   # shape [BLOCK_N]
    offs_m = col_start + tl.arange(0, BLOCK_M)   # shape [BLOCK_M]

    mask_n = offs_n < N
    mask_m = offs_m < M
    mask = mask_n[:, None] & mask_m[None, :]     # shape [BLOCK_N, BLOCK_M]

    # compute pointers for 2D load/store; assume row-major layout
    x_ptrs = x_ptr + offs_n[:, None] * stride_x + offs_m[None, :]
    y_ptrs = y_ptr + offs_n[:, None] * stride_y + offs_m[None, :]
    x_vals = tl.load(x_ptrs, mask=mask, other=0.0)
    y_vals = tl.load(y_ptrs, mask=mask, other=0.0)

    out_vals = (x_vals + y_vals) * y_vals

    out_ptrs = out_ptr + offs_n[:, None] * stride_out + offs_m[None, :]
    tl.store(out_ptrs, out_vals, mask=mask)

def triton_square(x: torch.Tensor):
    """
    Square each element of x using a Triton kernel.
    """
    assert x.is_cuda, "Input must be a CUDA tensor."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    # grid depending on block size chosen by autotune
    grid = lambda meta: ((n_elements + meta["BLOCK"] - 1) // meta["BLOCK"],)
    _square_kernel[grid](x, out, n_elements)
    return out

def triton_fused_addmul(x: torch.Tensor, y: torch.Tensor):
    """
    Compute out = (x + y) * y using a fused 2D Triton kernel (no intermediate allocation).
    Inputs are expected to be 2D tensors of shape (N, M), row-major contiguous.
    """
    assert x.is_cuda and y.is_cuda, "Inputs must be CUDA tensors."
    assert x.shape == y.shape, "Inputs must have the same shape."
    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x)
    N, M = x.shape
    # stride over rows (number of elements to advance to next row)
    stride_x = x.stride(0)
    stride_y = y.stride(0)
    stride_out = out.stride(0)
    grid = lambda meta: ((N + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],
                         (M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"])
    _fused_add_mul_kernel[grid](x, y, out, M, N, stride_x, stride_y, stride_out)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that preserves the semantics of the original Model by using equivalent
    PyTorch modules. This ensures correctness even if a prior specialization was incorrect.
    """
    def __init__(self, in_features=None, out_features=None, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        # Create layers matching the original Model so outputs match exactly.
        assert in_features is not None and out_features is not None, "in_features and out_features must be provided"
        self.bmm = nn.Linear(in_features, out_features)
        self.instance_norm = nn.InstanceNorm2d(out_features, eps=eps, momentum=momentum)

    def forward(self, x, y):
        x = self.bmm(x)
        x = self.instance_norm(x.unsqueeze(1).unsqueeze(1)).squeeze(1).squeeze(1)
        # Fuse add and mul in a single Triton kernel to avoid materializing the intermediate tensor.
        out = triton_fused_addmul(x, y)
        return out


# Keep the same input generation helpers for compatibility with harnesses
batch_size = 1024
in_features = 8192
out_features = 8192

def get_inputs():
    # Note: the harness is expected to move tensors to CUDA if needed.
    return [torch.rand(batch_size, in_features), torch.rand(batch_size, out_features)]

def get_init_inputs():
    return [in_features, out_features]