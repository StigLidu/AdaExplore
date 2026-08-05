import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configurations tuned for NVIDIA A6000 / Ampere.
# Favor large tiles that make good use of Tensor Cores for large M=N=8192.
MATMUL_AUTOTUNE = [
    triton.Config({"BLOCK_M": 512, "BLOCK_N": 512, "BLOCK_K": 128}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 512, "BLOCK_N": 512, "BLOCK_K": 256}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 512, "BLOCK_K": 256}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 512, "BLOCK_N": 256, "BLOCK_K": 256}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 128}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64},  num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},  num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64},  num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 512, "BLOCK_K": 64},  num_warps=8, num_stages=3),
]


@triton.autotune(configs=MATMUL_AUTOTUNE, key=['M', 'N', 'K'])
@triton.jit
def _matmul_fused_epilogue_kernel(A_ptr, B_ptr, C_ptr,
                                  M, N, K,
                                  stride_am, stride_ak,
                                  stride_bk, stride_bn,
                                  stride_cm, stride_cn,
                                  bias_ptr,
                                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """
    Fused tiled GEMM with fp16 inputs and fp32 accumulation, followed by epilogue:
      s = sigmoid(acc)
      y = 0.5 * acc * s
      y = clamp(y, -1, 1)
      t = tanh(y) via identity 2*sigmoid(2*y)-1
      t = clamp(t, -1, 1)
    A: (M, K) row-major fp16
    B: (K, N) row-major fp16
    C: (M, N) row-major fp32 (output)
    bias_ptr: (N,) fp32
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # iterate along K
    for k in range(0, K, BLOCK_K):
        k_block = tl.arange(0, BLOCK_K)

        a_ptrs = A_ptr + (offs_m[:, None] * stride_am +
                          (k + k_block)[None, :] * stride_ak)
        b_ptrs = B_ptr + ((k + k_block)[:, None] * stride_bk +
                          offs_n[None, :] * stride_bn)

        a_mask = (offs_m[:, None] < M) & ((k + k_block)[None, :] < K)
        b_mask = ((k + k_block)[:, None] < K) & (offs_n[None, :] < N)

        # load fp16 tiles (will be promoted/handled by Triton for dot)
        A_block = tl.load(a_ptrs, mask=a_mask, other=0.0)
        B_block = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # accumulate (tl.dot on fp16 produces fp32 accumulation)
        acc += tl.dot(A_block, B_block)

    # add bias (broadcast over M)
    bias = tl.load(bias_ptr + offs_n, mask=(offs_n < N), other=0.0)
    acc = acc + bias[None, :]

    # Epilogue:
    # sigmoid(acc)
    s = 1.0 / (1.0 + tl.exp(-acc))
    # y = 0.5 * acc * s (swish then divide by 2)
    y = 0.5 * acc * s
    # clamp prior to tanh to match semantics & stabilize exp
    y = tl.minimum(tl.maximum(y, -1.0), 1.0)
    # tanh via identity: 2*sigmoid(2*y) - 1
    t = 2.0 * (1.0 / (1.0 + tl.exp(-2.0 * y))) - 1.0
    # final clamp (tanh already in [-1,1], but keep to preserve original behavior)
    t = tl.minimum(tl.maximum(t, -1.0), 1.0)

    # store
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    store_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, t, mask=store_mask)


def triton_fused_linear(x: torch.Tensor, linear: nn.Linear,
                        weight_t_half: torch.Tensor = None,
                        bias_fp32: torch.Tensor = None) -> torch.Tensor:
    """
    Fused Linear (x @ W.T + bias) with epilogue using Triton.
    - x: (M, K) fp32 on CUDA (will be converted to fp16)
    - weight_t_half: optional cached (K, N) fp16 on CUDA (transposed weight)
    - bias_fp32: optional cached (N,) fp32 on CUDA
    Returns (M, N) fp32 on CUDA.
    """
    assert x.is_cuda and x.dtype == torch.float32, "x must be a float32 CUDA tensor"

    device = x.device

    # Prepare transposed fp16 weight on device
    if weight_t_half is None:
        weight_t = linear.weight.t().contiguous().half().to(device)
    else:
        weight_t = weight_t_half if weight_t_half.device == device else weight_t_half.to(device)

    # Convert input to fp16 once (contiguous) to reduce per-kernel overhead
    A = x.contiguous().half().to(device)

    # Prepare bias in fp32 on device
    if bias_fp32 is None:
        if linear.bias is None:
            bias = torch.zeros(weight_t.shape[1], device=device, dtype=torch.float32)
        else:
            bias = linear.bias.contiguous().to(torch.float32).to(device)
    else:
        bias = bias_fp32 if bias_fp32.device == device else bias_fp32.to(device)

    M, K = A.shape
    K_, N = weight_t.shape
    assert K == K_, f"Inner dims mismatch: {K} vs {K_}"

    # Output in fp32
    out = torch.empty((M, N), device=device, dtype=torch.float32)

    # Row-major contiguous strides
    stride_am = K
    stride_ak = 1
    stride_bk = N
    stride_bn = 1
    stride_cm = N
    stride_cn = 1

    grid = lambda meta: ((M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
                         (N + meta["BLOCK_N"] - 1) // meta["BLOCK_N"])

    _matmul_fused_epilogue_kernel[grid](A, weight_t, out,
                                        M, N, K,
                                        stride_am, stride_ak,
                                        stride_bk, stride_bn,
                                        stride_cm, stride_cn,
                                        bias)
    return out


class ModelNew(nn.Module):
    """
    Optimized Model:
      - Fuses Linear (GEMM) and the elementwise epilogue (swish -> /2 -> clamp -> tanh -> clamp)
        into a single Triton kernel to minimize memory traffic.
      - Uses mixed precision: fp16 matmul with fp32 accumulation and fp32 epilogue.
      - Caches device-side transposed fp16 weight and fp32 bias to avoid repeated host->device copies.
      - Falls back to original PyTorch ops when input is not on CUDA.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self._weight_t_half = None
        self._bias_fp32 = None
        # Track the original parameter pointer to detect weight changes (e.g., training updates)
        self._cached_weight_data_ptr = None

    def _ensure_device_cache(self, device):
        # Recreate caches if missing, on different device, or if weights changed
        weight_ptr = self.gemm.weight.data_ptr()
        need_recache = False
        if self._weight_t_half is None or self._weight_t_half.device != device:
            need_recache = True
        if self._cached_weight_data_ptr is None or self._cached_weight_data_ptr != weight_ptr:
            need_recache = True

        if need_recache:
            # create device-local cached copies
            self._weight_t_half = self.gemm.weight.t().contiguous().half().to(device)
            if self.gemm.bias is not None:
                self._bias_fp32 = self.gemm.bias.contiguous().to(torch.float32).to(device)
            else:
                self._bias_fp32 = torch.zeros(self.gemm.out_features, device=device, dtype=torch.float32)
            self._cached_weight_data_ptr = weight_ptr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CPU fallback uses native PyTorch ops to preserve exact semantics
        if not x.is_cuda:
            y = self.gemm(x)
            y = y * torch.sigmoid(y)  # Swish
            y = y / 2.0
            y = torch.clamp(y, min=-1.0, max=1.0)
            y = torch.tanh(y)
            y = torch.clamp(y, min=-1.0, max=1.0)
            return y

        device = x.device
        self._ensure_device_cache(device)
        return triton_fused_linear(x, self.gemm, self._weight_t_half, self._bias_fp32)


# Keep same interface/constants as original model for compatibility
batch_size = 1024
in_features = 8192
out_features = 8192

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features]