import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configurations tuned for A6000 / Ampere
# Added larger BLOCK_K candidates and some larger tiles to better utilize Tensor Cores on Ampere.
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},  num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32},  num_warps=8, num_stages=3),

    # Larger BLOCK_K options for improved compute:memory and Tensor Core utilization
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},  num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64},  num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64},  num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64},  num_warps=8, num_stages=3),

    # A few larger candidates with bigger BLOCK_M to exercise different tilings
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8, num_stages=4),
]


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def _fused_gemm_bias_relu(
    A_ptr,            # A pointer (matrix) (M, K)
    B_ptr,            # B pointer (matrix) (K, N)  <-- we will pass weight.t() so B is (K,N)
    bias_ptr,         # bias pointer (N,)
    C_ptr,            # output pointer (M, N)
    M, N, K,          # matrix sizes (ints)
    stride_am, stride_ak,   # A strides (in elements)
    stride_bk, stride_bn,   # B strides (in elements)
    stride_cm, stride_cn,   # C/output strides (in elements)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """
    Tiled GEMM kernel with bias add and ReLU fused.
    Computes C = relu(A @ B + bias)
    A: (M, K)
    B: (K, N)
    bias: (N,)
    C: (M, N)
    """

    # Block indices
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)

    # Offsets for this block
    row_off = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    col_off = col_block * BLOCK_N + tl.arange(0, BLOCK_N)

    # Create accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k_start in range(0, K, BLOCK_K):
        k_off = k_start + tl.arange(0, BLOCK_K)

        # Compute addresses for loads
        A_addresses = A_ptr + (row_off[:, None] * stride_am + k_off[None, :] * stride_ak)
        B_addresses = B_ptr + (k_off[:, None] * stride_bk + col_off[None, :] * stride_bn)

        # Masks for valid loads
        mask_a = (row_off[:, None] < M) & (k_off[None, :] < K)
        mask_b = (k_off[:, None] < K) & (col_off[None, :] < N)

        # Load tiles. On the host we will pass A and B as fp16 to leverage Tensor Cores;
        # here we load (which yields fp16) and then cast to fp32 for accumulation.
        a = tl.load(A_addresses, mask=mask_a, other=0.0)
        b = tl.load(B_addresses, mask=mask_b, other=0.0)

        # Cast to FP32 for accumulation (preserve numeric stability while enabling Tensor Cores on fp16 inputs)
        a = tl.cast(a, tl.float32)
        b = tl.cast(b, tl.float32)

        # Accumulate (FP32)
        acc += tl.dot(a, b)

    # Add bias (load once per column)
    bias_vals = tl.load(bias_ptr + col_off, mask=col_off < N, other=0.0)  # shape [BLOCK_N]
    acc = acc + bias_vals[None, :]

    # Apply ReLU
    acc = tl.maximum(acc, 0.0)

    # Store results with mask
    C_addresses = C_ptr + (row_off[:, None] * stride_cm + col_off[None, :] * stride_cn)
    mask_c = (row_off[:, None] < M) & (col_off[None, :] < N)
    tl.store(C_addresses, acc, mask=mask_c)


def triton_gemm_bias_relu(A: torch.Tensor, B_t: torch.Tensor, bias: torch.Tensor):
    """
    Wrapper to call the Triton fused kernel.

    Host-side responsibilities for mixed-precision:
    - Convert A and B_t to fp16 on the host so the kernel can leverage Tensor Cores.
    - Keep bias as fp32 so after FP32 accumulation we can add bias in fp32 and store fp32 outputs.
    A: (M, K) contiguous float32 or float16 on CUDA (we will ensure fp16)
    B_t: (K, N) contiguous float32 or float16 on CUDA (we will ensure fp16)
    bias: (N,) contiguous float32 on CUDA
    returns: C (M, N) float32 CUDA tensor
    """
    assert A.is_cuda and B_t.is_cuda and bias.is_cuda, "All tensors must be CUDA tensors."

    # Convert inputs to fp16 for Tensor Core throughput (do this on host to avoid extra kernel work).
    # If tensors are already half, this will still produce a contiguous view/copy as needed.
    A_h = A.contiguous().half()
    B_h = B_t.contiguous().half()
    bias_c = bias.contiguous().float()  # ensure bias is fp32

    M, K = A_h.shape
    K2, N = B_h.shape
    assert K == K2, "Inner dimensions must match."

    # Output tensor (we accumulate in fp32 and store fp32)
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)

    # Strides in elements (half tensors have same element stride logic)
    stride_am = A_h.stride(0)
    stride_ak = A_h.stride(1)
    stride_bk = B_h.stride(0)
    stride_bn = B_h.stride(1)
    stride_cm = C.stride(0)
    stride_cn = C.stride(1)

    # Grid
    def grid(meta):
        return (
            (M + meta['BLOCK_M'] - 1) // meta['BLOCK_M'],
            (N + meta['BLOCK_N'] - 1) // meta['BLOCK_N'],
        )

    _fused_gemm_bias_relu[grid](
        A_h, B_h, bias_c, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn
    )

    return C


class ModelNew(nn.Module):
    """
    Optimized model that fuses GEMM + bias + ReLU using a Triton kernel.
    """
    def __init__(self, in_features, out_features, bias_shape):
        super(ModelNew, self).__init__()
        # Keep a regular Linear module to store weights; we'll transpose at runtime for the kernel
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        # Bias parameter
        self.bias = nn.Parameter(torch.randn(bias_shape))

        # Cached transposed weight in (K, N) layout stored as fp16 to avoid per-forward transpose/copy.
        # We register a buffer so it moves with the module and can be inspected by the user.
        self.register_buffer("weight_t", torch.empty(0, dtype=torch.float16))
        # Track the data_ptr of the source weight to detect weight updates
        self._w_ptr = None

    def forward(self, x):
        """
        Forward performs fused matrix multiply, bias add, and ReLU via Triton kernel.
        x: (batch_size, in_features)
        returns: (batch_size, out_features)
        """
        device = x.device

        # Ensure the module parameters are on the same device as input.
        if self.gemm.weight.device != device:
            self.gemm.weight.data = self.gemm.weight.data.to(device)
        if self.bias.device != device:
            self.bias.data = self.bias.data.to(device)

        # Lazily update the cached transposed fp16 weight if the underlying weight changed
        # or the cache is empty or lives on a different device.
        try:
            weight_ptr = self.gemm.weight.data_ptr()
        except Exception:
            weight_ptr = None

        # If cache is empty, on a different device, or the weight memory changed, refresh it.
        if self.weight_t.numel() == 0 or self.weight_t.device != device or self._w_ptr != weight_ptr:
            # Create a contiguous transposed half tensor on the correct device
            self.weight_t = self.gemm.weight.t().contiguous().half().to(device)
            # Record the pointer so we can detect future changes to the weight storage
            self._w_ptr = weight_ptr

        # Call the fused Triton kernel (the wrapper will further ensure inputs are half as needed)
        out = triton_gemm_bias_relu(x, self.weight_t, self.bias)

        return out