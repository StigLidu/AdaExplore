import torch
import torch.nn as nn
import triton
import triton.language as tl

# ======================
# High-performance matmul kernel (fp16 inputs, fp32 accumulation)
# Tuned/autotuned for Ampere A6000 — returns fp32 with fp16 inputs supported.
# ======================
AUTOTUNE_MATMUL = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 512, "BLOCK_K": 64},  num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64},  num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 512, "BLOCK_K": 32},  num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 256, "BLOCK_K": 32},  num_warps=4, num_stages=1),
]

@triton.autotune(configs=AUTOTUNE_MATMUL, key=['M', 'N', 'K'])
@triton.jit
def _matmul_kernel(
    A_ptr,        # (M, K) row-major
    B_ptr,        # (K, N) row-major (B transposed when caller passes B_t)
    C_ptr,        # (M, N) row-major
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row_off = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_off = k0 + tl.arange(0, BLOCK_K)

        a_ptrs = A_ptr + (row_off[:, None] * stride_am) + (k_off[None, :] * stride_ak)
        b_ptrs = B_ptr + (k_off[:, None] * stride_bk) + (col_off[None, :] * stride_bn)

        a_mask = (row_off[:, None] < M) & (k_off[None, :] < K)
        b_mask = (k_off[:, None] < K) & (col_off[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

    c_ptrs = C_ptr + (row_off[:, None] * stride_cm) + (col_off[None, :] * stride_cn)
    c_mask = (row_off[:, None] < M) & (col_off[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_matmul(A: torch.Tensor, B_t: torch.Tensor):
    """
    Compute A @ B where B_t is B.T using Triton matmul kernel.
    Accepts fp16 or fp32 inputs; returns fp32 output.
    """
    assert A.is_cuda and B_t.is_cuda, "Inputs must be on CUDA"
    A = A.contiguous()
    B_t = B_t.contiguous()

    M, K = A.shape
    K2, N = B_t.shape
    assert K == K2, "Incompatible shapes for matmul"

    device = A.device
    C = torch.empty((M, N), device=device, dtype=torch.float32)

    # strides for row-major
    stride_am = K
    stride_ak = 1
    stride_bk = N
    stride_bn = 1
    stride_cm = N
    stride_cn = 1

    grid = lambda meta: ((M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
                         (N + meta["BLOCK_N"] - 1) // meta["BLOCK_N"])

    _matmul_kernel[grid](
        A, B_t, C, M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn
    )
    return C


# ======================
# Fused bimatrix+tanh kernel:
# Computes out = tanh(A1 @ B1 + A2 @ B2 + bias) with fp16 inputs and fp32 accumulation.
# A1: (M, K1), B1_t: (K1, N)
# A2: (M, K2), B2_t: (K2, N)
# Returns fp32 output (M, N)
# ======================
AUTOTUNE_BIMATMUL = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 512, "BLOCK_K": 64},  num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64},  num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 512, "BLOCK_K": 32},  num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 256, "BLOCK_K": 32},  num_warps=4, num_stages=1),
]

@triton.autotune(configs=AUTOTUNE_BIMATMUL, key=['M', 'N', 'K1', 'K2'])
@triton.jit
def _bimatrix_tanh_kernel(
    A1_ptr, A2_ptr, B1_ptr, B2_ptr, bias_ptr, C_ptr,
    M, N, K1, K2,
    stride_a1m, stride_a1k,
    stride_a2m, stride_a2k,
    stride_b1k, stride_b1n,
    stride_b2k, stride_b2n,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row_off = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # First product: A1 @ B1
    for k0 in range(0, K1, BLOCK_K):
        k_off = k0 + tl.arange(0, BLOCK_K)
        a1_ptrs = A1_ptr + (row_off[:, None] * stride_a1m) + (k_off[None, :] * stride_a1k)
        b1_ptrs = B1_ptr + (k_off[:, None] * stride_b1k) + (col_off[None, :] * stride_b1n)

        a1_mask = (row_off[:, None] < M) & (k_off[None, :] < K1)
        b1_mask = (k_off[:, None] < K1) & (col_off[None, :] < N)

        a1 = tl.load(a1_ptrs, mask=a1_mask, other=0.0)
        b1 = tl.load(b1_ptrs, mask=b1_mask, other=0.0)
        acc += tl.dot(a1, b1)

    # Second product: A2 @ B2
    for k0 in range(0, K2, BLOCK_K):
        k_off = k0 + tl.arange(0, BLOCK_K)
        a2_ptrs = A2_ptr + (row_off[:, None] * stride_a2m) + (k_off[None, :] * stride_a2k)
        b2_ptrs = B2_ptr + (k_off[:, None] * stride_b2k) + (col_off[None, :] * stride_b2n)

        a2_mask = (row_off[:, None] < M) & (k_off[None, :] < K2)
        b2_mask = (k_off[:, None] < K2) & (col_off[None, :] < N)

        a2 = tl.load(a2_ptrs, mask=a2_mask, other=0.0)
        b2 = tl.load(b2_ptrs, mask=b2_mask, other=0.0)
        acc += tl.dot(a2, b2)

    # add bias (broadcast across rows)
    col_mask = col_off < N
    bias_vals = tl.load(bias_ptr + col_off, mask=col_mask, other=0.0)
    acc = acc + bias_vals[None, :]

    # tanh via exp to avoid calling unavailable tanh
    exp2 = tl.exp(acc * 2.0)
    acc = (exp2 - 1.0) / (exp2 + 1.0)

    # store fp32 hidden result
    c_ptrs = C_ptr + (row_off[:, None] * stride_cm) + (col_off[None, :] * stride_cn)
    c_mask = (row_off[:, None] < M) & (col_off[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_bimatrix_tanh(A1: torch.Tensor, A2: torch.Tensor, B1_t: torch.Tensor, B2_t: torch.Tensor, bias: torch.Tensor):
    """
    Compute tanh(A1 @ B1 + A2 @ B2 + bias) using a fused Triton kernel.
    Inputs A1,A2,B1_t,B2_t typically fp16 for best throughput; returns fp32 tensor.
    """
    assert A1.is_cuda and A2.is_cuda and B1_t.is_cuda and B2_t.is_cuda and bias.is_cuda, "All tensors must be CUDA"
    A1 = A1.contiguous()
    A2 = A2.contiguous()
    B1_t = B1_t.contiguous()
    B2_t = B2_t.contiguous()
    bias = bias.contiguous()

    M, K1 = A1.shape
    M2, K2 = A2.shape
    assert M == M2, "A1 and A2 must have the same M"
    K1_b, N = B1_t.shape
    K2_b, N2 = B2_t.shape
    assert K1 == K1_b and K2 == K2_b and N == N2, "Incompatible shapes"

    device = A1.device
    C = torch.empty((M, N), device=device, dtype=torch.float32)

    stride_a1m = K1
    stride_a1k = 1
    stride_a2m = K2
    stride_a2k = 1
    stride_b1k = N
    stride_b1n = 1
    stride_b2k = N
    stride_b2n = 1
    stride_cm = N
    stride_cn = 1

    grid = lambda meta: ((M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
                         (N + meta["BLOCK_N"] - 1) // meta["BLOCK_N"])

    _bimatrix_tanh_kernel[grid](A1, A2, B1_t, B2_t, bias, C,
                                M, N, K1, K2,
                                stride_a1m, stride_a1k,
                                stride_a2m, stride_a2k,
                                stride_b1k, stride_b1n,
                                stride_b2k, stride_b2n,
                                stride_cm, stride_cn)
    return C


# ======================
# Optimized ModelNew:
# - Fuses the two i2h GEMMs + bias + tanh into a single Triton kernel (triton_bimatrix_tanh).
# - Uses an optimized Triton matmul for h2o.
# - Aggressive caching of transposed fp16 weights to minimize CPU/GPU overhead.
# - Minimizes temporary allocations and device transfers.
# ======================
class ModelNew(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # preserve original persistent hidden initialization semantics
        # (module-level batch_size is expected to be defined in the module globals)
        self.hidden = torch.randn((batch_size, hidden_size))

        # linear layers to preserve API semantics
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

        # caches for transposed fp16 weights (on-device). Keep pointers to detect changes.
        self._i2h_w_ptr = None
        self._i2h_w_x_t_fp16 = None
        self._i2h_w_h_t_fp16 = None

        self._h2o_w_ptr = None
        self._h2o_w_t_fp16 = None

    def _ensure_i2h_cache(self, device):
        i2h_w_ptr = self.i2h.weight.data_ptr()
        if (self._i2h_w_ptr != i2h_w_ptr) or (self._i2h_w_x_t_fp16 is None) or (self._i2h_w_x_t_fp16.device != device):
            W = self.i2h.weight  # (hidden_size, input_size + hidden_size)
            W_x = W[:, : self.input_size]    # (hidden_size, input_size)
            W_h = W[:, self.input_size :]    # (hidden_size, hidden_size)
            # store transposed slices as fp16 contiguous on-device for TensorCore-friendly GEMMs
            self._i2h_w_x_t_fp16 = W_x.t().contiguous().to(device).half()
            self._i2h_w_h_t_fp16 = W_h.t().contiguous().to(device).half()
            self._i2h_w_ptr = i2h_w_ptr

    def _ensure_h2o_cache(self, device):
        h2o_w_ptr = self.h2o.weight.data_ptr()
        if (self._h2o_w_ptr != h2o_w_ptr) or (self._h2o_w_t_fp16 is None) or (self._h2o_w_t_fp16.device != device):
            self._h2o_w_t_fp16 = self.h2o.weight.t().contiguous().to(device).half()
            self._h2o_w_ptr = h2o_w_ptr

    def forward(self, x: torch.Tensor, initial_hidden: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass:
          - Optionally copy initial_hidden into persistent hidden (preserve reference model behavior)
          - Compute hidden_next = tanh(x @ W_x^T + hidden @ W_h^T + bias_i2h) using fused Triton kernel
          - Compute output = hidden_next @ W_o^T + bias_h2o via optimized Triton matmul
          - Keep operations in fp16 for GEMMs (fp32 accumulation), results are fp32
        """
        assert x.dtype == torch.float32, "Only float32 inputs supported"

        if initial_hidden is not None:
            # preserve original semantics: copy into persistent hidden
            self.hidden.copy_(initial_hidden)

        # move persistent hidden to input device
        self.hidden = self.hidden.to(x.device)
        device = x.device

        # refresh cached transposed weights if needed
        self._ensure_i2h_cache(device)
        self._ensure_h2o_cache(device)

        # Convert inputs to fp16 once (minimize repeated casting overhead)
        x_fp16 = x.contiguous().to(device).half()
        hidden_fp16 = self.hidden.contiguous().to(device).half()

        # fused bimatrix + tanh -> returns fp32 hidden next
        bias_i2h = self.i2h.bias.to(device).contiguous().to(torch.float32)
        hidden_next = triton_bimatrix_tanh(
            x_fp16, hidden_fp16,
            self._i2h_w_x_t_fp16, self._i2h_w_h_t_fp16,
            bias_i2h
        )

        # update persistent hidden
        self.hidden = hidden_next

        # compute output: convert hidden_next to fp16 and run optimized matmul
        out = triton_matmul(hidden_next.half(), self._h2o_w_t_fp16)
        out = out + self.h2o.bias.unsqueeze(0).to(device)

        return out


# ======================
# Global sizes preserved for compatibility with original entrypoints
# ======================
batch_size = 256
input_size = 16384
hidden_size = 16384
output_size = 8192
sequence_length = 256

def get_inputs():
    # Return CUDA tensors (float32) for the model inputs
    return [torch.rand(batch_size, input_size, dtype=torch.float32).cuda(),
            torch.rand(batch_size, hidden_size, dtype=torch.float32).cuda()]

def get_init_inputs():
    return [input_size, hidden_size, output_size]