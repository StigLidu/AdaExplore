import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configs tuned for NVIDIA A6000 (Ampere).
# Added larger BLOCK_N candidates and some higher-warp configs to better utilize Tensor Cores
# and amortize launch overhead for very-wide layers in this model.
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256,  "BLOCK_K": 32},  num_warps=8,  num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 512,  "BLOCK_K": 32},  num_warps=8,  num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 512,  "BLOCK_K": 64},  num_warps=8,  num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 1024, "BLOCK_K": 64},  num_warps=8,  num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 2048, "BLOCK_K": 64},  num_warps=8,  num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 4096, "BLOCK_K": 64},  num_warps=16, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 1024, "BLOCK_K": 64},  num_warps=8,  num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 2048, "BLOCK_K": 64},  num_warps=16, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 4096, "BLOCK_K": 128}, num_warps=16, num_stages=3),
]

_ZERO_BIAS_CACHE = {}

@triton.autotune(
    configs=AUTOTUNE_CONFIGS,
    key=["M", "N", "K"],
)
@triton.jit
def _gemm_kernel(
    A_ptr,  # (M, K)
    B_ptr,  # (K, N)
    C_ptr,  # (M, N)
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    bias_ptr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    A_IS_FP16: tl.constexpr, B_IS_FP16: tl.constexpr, APPLY_BIAS: tl.constexpr, APPLY_RELU: tl.constexpr, OUT_IS_FP16: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offs = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # iterate over K dimension
    for kb_start in range(0, K, BLOCK_K):
        k_range = kb_start + k_offs

        a_ptrs = A_ptr + (row_offs[:, None] * stride_am) + (k_range[None, :] * stride_ak)
        b_ptrs = B_ptr + (k_range[:, None] * stride_bk) + (col_offs[None, :] * stride_bn)

        a_mask = (row_offs[:, None] < M) & (k_range[None, :] < K)
        b_mask = (k_range[:, None] < K) & (col_offs[None, :] < N)

        # Load A and B in their native dtype (allows Tensor Cores when both fp16)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Perform dot with correct casting paths to enable fp16-fp16 dot -> fp32 acc (Tensor Cores)
        if A_IS_FP16 and B_IS_FP16:
            acc += tl.dot(a, b)
        elif A_IS_FP16 and (not B_IS_FP16):
            acc += tl.dot(tl.cast(a, tl.float32), b)
        elif (not A_IS_FP16) and B_IS_FP16:
            acc += tl.dot(a, tl.cast(b, tl.float32))
        else:
            acc += tl.dot(a, b)

    if APPLY_BIAS:
        bias_ptrs = bias_ptr + col_offs[None, :]
        bias_mask = (col_offs[None, :] < N)
        bias_vals = tl.load(bias_ptrs, mask=bias_mask, other=0.0)
        acc += bias_vals

    if APPLY_RELU:
        acc = tl.maximum(acc, 0.0)

    # store - either fp32 or fp16 depending on OUT_IS_FP16
    c_ptrs = C_ptr + (row_offs[:, None] * stride_cm) + (col_offs[None, :] * stride_cn)
    c_mask = (row_offs[:, None] < M) & (col_offs[None, :] < N)

    if OUT_IS_FP16:
        tl.store(c_ptrs, tl.cast(acc, tl.float16), mask=c_mask)
    else:
        tl.store(c_ptrs, acc, mask=c_mask)


def triton_gemm(A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor = None, apply_relu: bool = False, out_fp16: bool = False):
    """
    A: (M, K) - torch.float32 or torch.float16
    B: (K, N) - torch.float32 or torch.float16 (weights are expected in row-major KxN)
    bias: (N,) float32 or None
    out_fp16: if True, output tensor is allocated and returned as torch.float16
    """
    assert A.is_cuda and B.is_cuda
    assert A.dtype in (torch.float32, torch.float16)
    assert B.dtype in (torch.float32, torch.float16)

    A_ = A if A.is_contiguous() else A.contiguous()
    B_ = B if B.is_contiguous() else B.contiguous()

    M, K = A_.shape
    Kb, N = B_.shape
    assert Kb >= K

    # Use padded Kb/Np for kernel grid convenience (kernel handles bounds)
    Kp = Kb
    Np = N

    out_dtype = torch.float16 if out_fp16 else torch.float32
    C = torch.empty((M, Np), device=A_.device, dtype=out_dtype)

    # compute strides (in elements)
    stride_am = A_.stride(0)
    stride_ak = A_.stride(1)
    stride_bk = B_.stride(0)
    stride_bn = B_.stride(1)
    stride_cm = C.stride(0)
    stride_cn = C.stride(1)

    apply_bias = bias is not None
    if apply_bias:
        bias_ = bias.contiguous()
        if bias_.numel() != Np:
            bias_pad = torch.zeros(Np, device=bias_.device, dtype=bias_.dtype)
            bias_pad[:bias_.numel()].copy_(bias_)
            bias_ = bias_pad
    else:
        key = (str(A_.device), Np)
        bias_ = _ZERO_BIAS_CACHE.get(key, None)
        if bias_ is None:
            bias_ = torch.zeros(Np, device=A_.device, dtype=torch.float32)
            _ZERO_BIAS_CACHE[key] = bias_

    a_is_fp16 = (A_.dtype == torch.float16)
    b_is_fp16 = (B_.dtype == torch.float16)

    def grid(meta):
        BM = meta["BLOCK_M"]
        BN = meta["BLOCK_N"]
        return ((M + BM - 1) // BM, (Np + BN - 1) // BN)

    _gemm_kernel[grid](
        A_, B_, C,
        M, Np, Kp,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        bias_,
        A_IS_FP16=a_is_fp16, B_IS_FP16=b_is_fp16, APPLY_BIAS=apply_bias, APPLY_RELU=apply_relu, OUT_IS_FP16=out_fp16,
    )

    if Np != N:
        C = C[:, :N]
    return C


class LinearTriton(nn.Module):
    """
    Linear layer backed by a Triton GEMM:
     - weight stored as (out_features, in_features) fp32 Parameter for optimizer stability
     - cached transposed fp16 contiguous buffer (K_pad x N_pad) as buffer for high throughput GEMM
     - supports lazy fp16 cache refresh
     - can output fp16 to chain hidden layers in lower precision for throughput
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, out_fp16: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.out_fp16 = out_fp16

        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float32))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float32))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        # padding strategy: pad K to multiples of 64 and N to multiples of 2048 to match autotune-friendly tiles
        _pad_k = 64
        _pad_n = 2048
        self._k_pad = ((in_features + _pad_k - 1) // _pad_k) * _pad_k
        self._n_pad = ((out_features + _pad_n - 1) // _pad_n) * _pad_n

        # create a padded transposed fp16 buffer (K_pad x N_pad) for efficient loads
        wt_pad = torch.zeros((self._k_pad, self._n_pad), dtype=torch.float16, device=self.weight.device)
        wt_view = self.weight.detach().t().contiguous().half()
        wt_pad[:wt_view.shape[0], :wt_view.shape[1]].copy_(wt_view)
        self.register_buffer('weight_t_fp16', wt_pad)

        # stamp for lazy refresh
        self._weight_version = self.weight._version

    def reset_parameters(self):
        fan_in = self.in_features
        bound = 1.0 / (fan_in ** 0.5)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)
        # keep fp16 cache in sync for freshly initialized params
        if hasattr(self, 'weight_t_fp16'):
            wt = self.weight.detach().t().contiguous().half()
            self.weight_t_fp16.zero_()
            self.weight_t_fp16[:wt.shape[0], :wt.shape[1]].copy_(wt)
            self._weight_version = self.weight._version

    def update_fp16_cache(self):
        if hasattr(self, 'weight_t_fp16'):
            wt = self.weight.detach().t().contiguous().half()
            # zero the padded buffer and copy valid region
            self.weight_t_fp16.zero_()
            self.weight_t_fp16[:wt.shape[0], :wt.shape[1]].copy_(wt)
            self._weight_version = self.weight._version

    def forward(self, x: torch.Tensor, apply_relu: bool = False):
        """
        x: (batch, in_features) - supports float32 or float16.
        If the layer is configured to output fp16 (out_fp16=True), the GEMM will write fp16 outputs.
        """
        # choose minimal-copy path for activations
        if self.out_fp16:
            if x.dtype == torch.float16:
                a = x if x.is_contiguous() else x.contiguous()
            else:
                a = x.half().contiguous()
        else:
            # want fp32 output; kernel supports mixed precision
            if x.dtype == torch.float32:
                a = x if x.is_contiguous() else x.contiguous()
            else:
                a = x if x.is_contiguous() else x.contiguous()

        # lazy refresh of fp16 weight cache if weight was changed
        if getattr(self, '_weight_version', None) != self.weight._version:
            self.update_fp16_cache()

        Wt_fp16 = self.weight_t_fp16  # shape (K_pad, N_pad)
        out = triton_gemm(a, Wt_fp16, bias=self.bias, apply_relu=apply_relu, out_fp16=self.out_fp16)
        return out


class ModelNew(nn.Module):
    """
    Optimized Model:
     - Replaces nn.Linear + ReLU with LinearTriton layers using fused bias and ReLU in the GEMM kernel.
     - Chains hidden layers in fp16 to maximize throughput while keeping final output in fp32.
     - Minimizes per-layer copies by pre-casting the input once when chaining fp16 hidden layers.
    """
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        super().__init__()
        layers = []
        current_in = input_size
        # Hidden layers: output in fp16 for throughput
        for i, hidden in enumerate(hidden_layer_sizes):
            layers.append(LinearTriton(current_in, hidden, bias=True, out_fp16=True))
            current_in = hidden
        # Final layer outputs fp32
        layers.append(LinearTriton(current_in, output_size, bias=True, out_fp16=False))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor):
        out = x
        num_layers = len(self.layers)

        # If any hidden layer produces fp16 (all but final), pre-cast once to fp16 and make contiguous
        hidden_fp16 = False
        if num_layers > 1:
            hidden_fp16 = any(getattr(l, "out_fp16", False) for l in list(self.layers)[:-1])

        if hidden_fp16:
            if out.dtype == torch.float32:
                out = out.half().contiguous()
            elif out.dtype == torch.float16 and not out.is_contiguous():
                out = out.contiguous()
        else:
            if not out.is_contiguous():
                out = out.contiguous()

        for i, layer in enumerate(self.layers):
            apply_relu = (i != num_layers - 1)
            out = layer(out, apply_relu=apply_relu)

        # ensure final output is float32
        if out.dtype == torch.float16:
            out = out.float()
        return out