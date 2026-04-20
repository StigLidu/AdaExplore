import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configurations for different block sizes
# Include a broad set of configurations (small -> large tiles) to let autotune pick
# high-throughput tilings for Ampere (A6000). These include wider 2D tiles and
# larger BLOCK_K options with more warps/stages for increased arithmetic intensity.
AUTOTUNE_CONFIGS = [
    # smaller configs for quick kernels / lower occupancy
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32},  num_warps=4,  num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},  num_warps=4,  num_stages=2),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 64,  "BLOCK_K": 32},  num_warps=8,  num_stages=2),

    # medium configs with larger BLOCK_K
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},  num_warps=8,  num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64},  num_warps=8,  num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128}, num_warps=16, num_stages=3),

    # larger 2D tiles for high throughput on Ampere
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64},  num_warps=16, num_stages=4),
    triton.Config({"BLOCK_M": 512, "BLOCK_N": 128, "BLOCK_K": 64},  num_warps=16, num_stages=4),
]


@triton.autotune(
    configs=AUTOTUNE_CONFIGS,
    key=["M", "N", "K"]
)
@triton.jit
def _fused_gemm_min_sub_kernel(
    a_ptr,            # pointer to A (batch x K)
    w_ptr,            # pointer to W (N x K)  (weight rows = output dim)
    bias_ptr,         # pointer to bias (N,) or 0
    out_ptr,          # pointer to output (batch x N)
    M,                # output rows (batch)
    N,                # output cols (out_features)
    K,                # reduction dim (in_features)
    a_stride_m,       # stride between rows of A
    a_stride_k,       # stride between cols of A
    w_stride_n,       # stride between rows of W (i.e., stride when increasing n)
    w_stride_k,       # stride between cols of W (i.e., stride when increasing k)
    out_stride_m,     # stride between rows of output
    out_stride_n,     # stride between cols of output
    const_val,        # scalar constant to apply min and subtraction
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program ids for row and col blocks
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Starting indices for this block
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    # Offsets within the block
    offs_m = m_start + tl.arange(0, BLOCK_M)  # (BLOCK_M,)
    offs_n = n_start + tl.arange(0, BLOCK_N)  # (BLOCK_N,)

    # Masks to guard out-of-bounds accesses
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks of BLOCK_K
    k = 0
    while k < K:
        k_range = k + tl.arange(0, BLOCK_K)  # (BLOCK_K,)

        # Load A block: shape (BLOCK_M, BLOCK_K)
        a_ptrs = a_ptr + (offs_m[:, None] * a_stride_m) + (k_range[None, :] * a_stride_k)
        a_mask = mask_m[:, None] & (k_range[None, :] < K)
        a_block = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load W block: we access W at [n, k], producing (BLOCK_K, BLOCK_N) when loading with k outer
        w_ptrs = w_ptr + (k_range[:, None] * w_stride_k) + (offs_n[None, :] * w_stride_n)
        w_mask = (k_range[:, None] < K) & mask_n[None, :]
        w_block = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Compute partial product and accumulate: a_block (M_block x K_block) dot w_block (K_block x N_block)
        acc += tl.dot(a_block, w_block)

        k += BLOCK_K

    # Add bias if provided (bias_ptr may be a valid pointer always in our wrapper)
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    # Apply elementwise min with constant, then subtract constant
    # Note: const_val is a scalar float passed as kernel argument
    acc = tl.minimum(acc, const_val)
    acc = acc - const_val

    # Store result back to output
    out_ptrs = out_ptr + (offs_m[:, None] * out_stride_m) + (offs_n[None, :] * out_stride_n)
    store_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptrs, acc, mask=store_mask)


def fused_gemm_min_sub(a: torch.Tensor, w: torch.Tensor, bias: torch.Tensor, const_val: float):
    """
    Wrapper that launches the Triton kernel to compute:
      out = min(a @ w (prepacked as (K, N)) + bias, const_val) - const_val

    Performance contract:
      - The kernel expects the weight to be prepacked in (K, N) contiguous layout and in float16.
        This enables unit-stride loads across N inside the kernel and allows Triton to use
        mixed-precision (fp16 inputs with fp32 accumulation) which is fast on Ampere GPUs.
      - The input 'a' should be float32; this wrapper will convert it to float16 for the kernel.
      - Bias is kept as float32 and added to the fp32 accumulator inside the kernel.
      - The output is produced as float32 (matching the original Model behavior).

    Assumes:
      - a: (M, K) contiguous, float32, CUDA
      - w: (K, N) contiguous, float16, CUDA (prepacked)
      - bias: (N,) contiguous or None, float32
    """
    assert a.is_cuda and w.is_cuda, "Inputs must be CUDA tensors"
    assert a.dtype == torch.float32, "Input A must be float32"
    assert w.dtype == torch.float16, "Weight must be prepacked as float16 with shape (K, N)"

    if bias is None:
        bias = torch.zeros(w.shape[1], device=w.device, dtype=torch.float32)
    else:
        bias = bias.contiguous().to(torch.float32)

    # Make sure A is contiguous and convert to fp16 for the kernel (we accumulate in fp32 inside the kernel)
    a = a.contiguous()
    a_half = a.half()

    # w is expected to be prepacked as (K, N) float16
    w_t = w  # name kept for clarity: w_t has shape (K, N) and dtype float16

    M, K = a.shape
    assert w_t.shape[0] == K, "Prepacked weight's first dim must equal K (in_features)"
    N = w_t.shape[1]

    # Prepare output (float32)
    out = torch.empty((M, N), device=a.device, dtype=torch.float32)

    # Strides for a_half and prepacked w_t
    a_stride_m = a_half.stride(0)
    a_stride_k = a_half.stride(1)
    # For w_t which is (K, N) contiguous, stride(0) corresponds to moving along K, stride(1) along N
    w_stride_k = w_t.stride(0)
    w_stride_n = w_t.stride(1)
    out_stride_m = out.stride(0)
    out_stride_n = out.stride(1)

    # Grid computed using autotune meta parameters
    def grid(meta):
        return ( (M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
                 (N + meta["BLOCK_N"] - 1) // meta["BLOCK_N"] )

    # Launch kernel: pass the fp16 prepacked weight and fp16 input; kernel accumulates in fp32 and writes float32 out
    _fused_gemm_min_sub_kernel[grid](
        a_half, w_t, bias, out,
        M, N, K,
        a_stride_m, a_stride_k,
        w_stride_n, w_stride_k,
        out_stride_m, out_stride_n,
        float(const_val),
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that fuses linear (matrix multiply + bias) with elementwise min and subtraction
    into a single Triton kernel for better GPU utilization and reduced memory traffic.
    """
    def __init__(self, in_features: int, out_features: int, constant: float):
        super(ModelNew, self).__init__()
        # Initialize weight and bias similar to nn.Linear
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        # constant as a learnable parameter to match original Model behavior (it used nn.Parameter)
        self.constant = nn.Parameter(torch.tensor(constant, dtype=torch.float32))
        # initialize parameters using the same scheme as nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5) if hasattr(torch, "sqrt") else 1.0)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0.0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor):
        # Ensure inputs are float32
        if x.dtype != torch.float32:
            x = x.to(dtype=torch.float32)

        if x.is_cuda:
            # Ensure we have a cached, prepacked weight in (K, N) float16 on the correct device.
            # We cache it as an attribute _weight_t to avoid doing the expensive transpose/copy on every forward.
            # Recompute if missing or if device changed.
            if not hasattr(self, "_weight_t") or self._weight_t is None or self._weight_t.device != x.device:
                # Pack weight to (K, N) float16 contiguous on the device of the input
                # self.weight has shape (N, K); we want (K, N)
                self._weight_t = self.weight.t().contiguous().half().to(x.device)

            # Ensure bias is on the same device (float32)
            bias_dev = self.bias.contiguous().to(x.device)

            # Call the fused kernel wrapper which expects the prepacked (K, N) float16 weight
            return fused_gemm_min_sub(x, self._weight_t, bias_dev, float(self.constant))
        else:
            # Fallback to CPU/PyTorch implementation
            out = torch.nn.functional.linear(x, self.weight, self.bias)
            out = torch.min(out, self.constant)
            out = out - self.constant
            return out


# Provide the same helper functions as the original module for convenience
batch_size = 128
in_features = 16384
out_features = 16384
constant = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, constant]


# Small import required for initialization math
import math