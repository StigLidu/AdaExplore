import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configurations tuned for NVIDIA A6000 (Ampere).
# Favor large tiles and larger BLOCK_K to better utilize Tensor Cores for K=8192.
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 128},  num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128},  num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128},  num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},   num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 256, "BLOCK_K": 128},  num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64},   num_warps=4, num_stages=2),
]


@triton.autotune(
    configs=AUTOTUNE_CONFIGS,
    key=['M', 'N', 'K']
)
@triton.jit
def fused_gemm_div_gelu_kernel(
    x_ptr,         # pointer to X (M, K)
    w_ptr,         # pointer to W (N, K)  (rows are output features)
    b_ptr,         # pointer to bias (N,)
    out_ptr,       # pointer to output (M, N)
    M, N, K,       # matrix sizes
    stride_xm, stride_xk,
    stride_wm, stride_wk,
    stride_b,
    stride_om, stride_on,
    divisor,       # scalar divisor (float)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """
    Compute out = GELU( (X @ W.T + b) / divisor )
    X: (M, K)
    W: (N, K)  (we access rows = output features)
    out: (M, N)
    Uses blocked matmul accumulating over K with mixed precision for throughput.
    GELU approximated with x * sigmoid(1.702 * x).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # base row and column indices for this program
    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)           # (BLOCK_M,)
    col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)           # (BLOCK_N,)

    # offsets along K for inner loop
    offs_k = tl.arange(0, BLOCK_K)                                  # (BLOCK_K,)

    # accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    while k < K:
        # compute masks for current k-chunk
        k_chunk = k + offs_k  # (BLOCK_K,)
        mask_k = k_chunk < K

        # pointers for X: shape (BLOCK_M, BLOCK_K)
        x_ptrs = x_ptr + (row_offsets[:, None] * stride_xm) + (k_chunk[None, :] * stride_xk)
        # pointers for W: shape (BLOCK_N, BLOCK_K)
        w_ptrs = w_ptr + (col_offsets[:, None] * stride_wm) + (k_chunk[None, :] * stride_wk)

        mask_x = (row_offsets[:, None] < M) & mask_k[None, :]
        mask_w = (col_offsets[:, None] < N) & mask_k[None, :]

        # load blocks (use other=0.0 to be explicit)
        x_block = tl.load(x_ptrs, mask=mask_x, other=0.0)   # (BLOCK_M, BLOCK_K)
        w_block = tl.load(w_ptrs, mask=mask_w, other=0.0)   # (BLOCK_N, BLOCK_K)

        # Cast to fp16 to leverage Tensor Cores; accumulation in fp32
        x_h = tl.cast(x_block, tl.float16)
        w_h = tl.cast(w_block, tl.float16)

        # accumulate: tl.dot does (BLOCK_M, BLOCK_K) x (BLOCK_K, BLOCK_N)
        acc += tl.dot(x_h, w_h.T)

        k += BLOCK_K

    # load bias
    bias_ptrs = b_ptr + col_offsets * stride_b
    bias = tl.load(bias_ptrs, mask=col_offsets < N, other=0.0)  # (BLOCK_N,)

    # add bias (broadcast over rows), divide, and apply GELU
    out_block = acc + bias[None, :]
    out_block = out_block / divisor

    # GELU approximation: x * sigmoid(1.702 * x)
    z = out_block
    sig = 1.0 / (1.0 + tl.exp(-1.702 * z))
    gelu = z * sig

    # store result
    out_ptrs = out_ptr + (row_offsets[:, None] * stride_om) + (col_offsets[None, :] * stride_on)
    mask_out = (row_offsets[:, None] < M) & (col_offsets[None, :] < N)
    tl.store(out_ptrs, gelu, mask=mask_out)


class ModelNew(nn.Module):
    """
    Optimized model that fuses the linear (matmul), division by a scalar, and GELU activation
    into a single Triton kernel for high throughput on large matrices.

    The forward computes:
        out = GELU( (x @ W.T + b) / divisor )

    Weight (self.linear) is kept as an nn.Linear for parameter management, but the kernel
    uses fp16 arithmetic for matrix multiply to exploit Tensor Cores and fp32 accumulation
    for numeric stability.
    """

    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, output_size, bias=True)
        # store divisor as float
        self.divisor = float(divisor)

    def forward(self, x: torch.Tensor):
        assert x.is_cuda, "Input must be on CUDA"
        # Keep a contiguous view of inputs
        x = x.contiguous()
        M, K = x.shape

        # Prepare weight and bias
        # The Linear weight is of shape (out_features, in_features) = (N, K)
        W = self.linear.weight
        b = self.linear.bias if self.linear.bias is not None else torch.zeros(W.shape[0], device=W.device, dtype=torch.float32)

        # Use fp16 view of weight to reduce memory traffic; ensure contiguous
        W_h = W.half().contiguous()
        b_f32 = b.contiguous().float()

        N = W_h.shape[0]

        # Prepare output in fp32
        out = torch.empty((M, N), device=x.device, dtype=torch.float32)

        # Strides (in elements)
        stride_xm, stride_xk = x.stride()
        stride_wm, stride_wk = W_h.stride()
        stride_b = b_f32.stride()[0] if b_f32.ndim > 0 else 1
        stride_om, stride_on = out.stride()

        # grid derived from autotune meta
        grid = lambda meta: (
            (M + meta['BLOCK_M'] - 1) // meta['BLOCK_M'],
            (N + meta['BLOCK_N'] - 1) // meta['BLOCK_N'],
        )

        # Launch fused Triton kernel
        fused_gemm_div_gelu_kernel[grid](
            x, W_h, b_f32, out,
            M, N, K,
            stride_xm, stride_xk,
            stride_wm, stride_wk,
            stride_b,
            stride_om, stride_on,
            float(self.divisor)
        )

        return out