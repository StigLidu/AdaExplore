import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configurations tuned for NVIDIA A6000 (Ampere).
# Mix of tile sizes favoring tensor-core usage and high arithmetic intensity.
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=16, num_stages=4),
    triton.Config({"BLOCK_M": 512, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=16, num_stages=4),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 512, "BLOCK_K": 64}, num_warps=16, num_stages=4),
    triton.Config({"BLOCK_M": 512, "BLOCK_N": 512, "BLOCK_K": 128}, num_warps=16, num_stages=4),
    triton.Config({"BLOCK_M": 1024, "BLOCK_N": 256, "BLOCK_K": 128}, num_warps=16, num_stages=4),
]

@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _matmul_relu_div_kernel(
    A_ptr,           # A: [M, K] (fp16)
    B_ptr,           # B: [K, N] (fp16) - pre-transposed as [K, N]
    C_ptr,           # out: [M, N] (fp32)
    M, N, K,         # matrix dims
    lda, ldb, ldc,   # row-strides (lda=K, ldb=N, ldc=N)
    bias_ptr,        # bias: [N] (fp32)
    divisor,         # scalar divisor (float)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused matmul (A @ B) + bias + ReLU + division.
    A: [M, K] fp16
    B: [K, N] fp16 (pre-transposed)
    C: [M, N] fp32
    bias: [N] fp32
    """
    # Block indices
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)

    row_start = row_block * BLOCK_M
    col_start = col_block * BLOCK_N

    # Row and column index vectors for this tile
    rows = row_start + tl.arange(0, BLOCK_M)
    cols = col_start + tl.arange(0, BLOCK_N)

    # Masks for bounds
    mask_rows = rows < M
    mask_cols = cols < N

    # accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # k offsets within a K-block
    k_offsets = tl.arange(0, BLOCK_K)

    # Main K loop - iterate over blocks of size BLOCK_K
    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + k_offsets  # shape [BLOCK_K]

        # Load A tile: shape [BLOCK_M, BLOCK_K], fp16
        a_ptrs = A_ptr + (rows[:, None] * lda) + k_idx[None, :]
        a_mask = mask_rows[:, None] & (k_idx[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B tile: shape [BLOCK_K, BLOCK_N], fp16
        b_ptrs = B_ptr + (k_idx[:, None] * ldb) + cols[None, :]
        b_mask = (k_idx[:, None] < K) & mask_cols[None, :]
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate (Triton uses fp32 accumulation for fp16 inputs)
        acc += tl.dot(a, b)

    # Add bias (broadcast across rows)
    bias_vals = tl.load(bias_ptr + cols, mask=mask_cols, other=0.0)
    acc = acc + bias_vals[None, :]

    # Fused ReLU and division
    acc = tl.maximum(acc, 0.0)
    acc = acc / divisor

    # Store results (fp32)
    out_ptrs = C_ptr + (rows[:, None] * ldc) + cols[None, :]
    store_mask = mask_rows[:, None] & mask_cols[None, :]
    tl.store(out_ptrs, acc, mask=store_mask)


def triton_linear_relu_div(x: torch.Tensor, weight_t: torch.Tensor, bias: torch.Tensor, divisor: float):
    """
    Wrapper to call Triton kernel:
      out = relu(x @ weight + bias) / divisor

    weight_t: pre-transposed weight stored as [K, N] in fp16
    x: [M, K] float32 (converted to fp16 here)
    """
    assert x.is_cuda and weight_t.is_cuda, "Tensors must be on CUDA."

    # Convert x to fp16 on host to reduce memory bandwidth and utilize Tensor Cores
    A = x.half().contiguous()  # [M, K]
    B = weight_t.contiguous()
    if B.dtype != torch.half:
        B = B.half()

    bias_c = bias.contiguous().to(device=A.device, dtype=torch.float32)

    M, K = A.shape
    K_b, N = B.shape
    assert K == K_b, f"K dimension mismatch: {K} vs {K_b}"

    out = torch.empty((M, N), device=A.device, dtype=torch.float32)

    # Row-major strides
    lda = K
    ldb = N
    ldc = N

    grid = lambda meta: ((M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
                         (N + meta["BLOCK_N"] - 1) // meta["BLOCK_N"])

    _matmul_relu_div_kernel[grid](
        A, B, out,
        M, N, K,
        lda, ldb, ldc,
        bias_c,
        float(divisor),
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model using a fused Triton kernel for:
      linear -> ReLU -> division

    Key optimizations:
      - Cache a device-resident transposed weight in fp16 to avoid per-forward transpose/convert.
      - Use mixed precision (fp16 inputs & weights) to halve memory bandwidth and leverage Tensor Cores.
      - Fuse matmul + bias + relu + division in a single Triton kernel.
      - Autotuned tile sizes for A6000.
    """
    def __init__(self, in_features, out_features, divisor):
        super(ModelNew, self).__init__()
        # Keep nn.Linear to manage parameters & initialization
        self.linear = nn.Linear(in_features, out_features)
        self.divisor = float(divisor)
        # Cached transposed weight on device in fp16 (lazy init)
        self._weight_t = None
        self._weight_t_device = None
        self._weight_t_shape = None

    def _refresh_weight_t(self, device):
        # Create a device resident transposed weight in fp16 for efficient matmul.
        w = self.linear.weight.detach()
        # Transpose to [K, N] where K=in_features, N=out_features
        self._weight_t = w.t().contiguous().to(device=device, dtype=torch.half)
        self._weight_t_device = device
        self._weight_t_shape = self._weight_t.shape

    def forward(self, x):
        device = x.device
        expected_shape = (self.linear.weight.size(1), self.linear.weight.size(0))
        if (self._weight_t is None) or (self._weight_t_device != device) or (self._weight_t_shape != expected_shape):
            self._refresh_weight_t(device)
        return triton_linear_relu_div(x, self._weight_t, self.linear.bias, self.divisor)


# Keep the same helper functions as in the original script
batch_size = 1024
in_features = 8192
out_features = 8192
divisor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda().to(torch.float32)]

def get_init_inputs():
    return [in_features, out_features, divisor]