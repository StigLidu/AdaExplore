import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configurations tuned for NVIDIA A6000 (Ampere) and large GEMMs.
AUTOTUNE_CONFIGS = [
    # Smaller K-tiles (32, 64) for better TensorCore utilization on Ampere
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 512, "BLOCK_K": 32}, num_warps=8, num_stages=4),

    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 512, "BLOCK_K": 64}, num_warps=8, num_stages=4),

    # Keep some larger K-tiles as fallbacks
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 512, "BLOCK_K": 128}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 512, "BLOCK_N": 512, "BLOCK_K": 128}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 128}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 512, "BLOCK_K": 128}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 256, "BLOCK_K": 128}, num_warps=4, num_stages=2),
]

@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def _fused_gemm_activation_fp16(
    A_ptr,        # (M, K) float16 or float32 (we'll pass float16 on fast path)
    B_ptr,        # (K, N) float16
    C_ptr,        # (M, N) float32 output
    M, N, K,
    stride_a0, stride_a1,
    stride_b0, stride_b1,
    stride_c0, stride_c1,
    bias_ptr,     # (N,) float32
    scale,        # float32
    hmin,         # float32
    hmax,         # float32
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Compute C = GELU(Hardtanh((A @ B) + bias) * scale)
    A: (M, K) float16/float32, B: (K, N) float16, bias: (N,) float32
    Output C is float32 for numerical stability.
    Uses blocked matmul with float32 accumulation and applies scaling+hardtanh+GELU inline.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for rows and columns this program will compute
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offsets < M
    n_mask = n_offsets < N

    # Accumulator in float32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in tiles of BLOCK_K
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # A tile pointers and load as float16/float32; tl.load returns element dtype
        a_ptrs = A_ptr + (m_offsets[:, None] * stride_a0) + (k_offsets[None, :] * stride_a1)
        a_mask = m_mask[:, None] & k_mask[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # B tile pointers: B is (K, N)
        b_ptrs = B_ptr + (k_offsets[:, None] * stride_b0) + (n_offsets[None, :] * stride_b1)
        b_mask = k_mask[:, None] & n_mask[None, :]
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Promote to float32 accumulator if needed via tl.dot
        acc += tl.dot(a, b)

    # Add bias if provided
    if bias_ptr is not None:
        bias_vals = tl.load(bias_ptr + n_offsets, mask=n_mask, other=0.0)
        acc = acc + bias_vals[None, :]

    # Apply scaling
    acc = acc * scale

    # Hardtanh clamp
    acc = tl.minimum(tl.maximum(acc, hmin), hmax)

    # GELU approximation using sigmoid-like formula: x * sigmoid(1.702 * x)
    z = 1.702 * acc
    sig = 1.0 / (1.0 + tl.exp(-z))
    acc = acc * sig

    # Store results back to C (float32)
    c_ptrs = C_ptr + (m_offsets[:, None] * stride_c0) + (n_offsets[None, :] * stride_c1)
    store_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(c_ptrs, acc, mask=store_mask)


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def _fused_gemm_activation_fp32(
    A_ptr,        # (M, K) float32
    B_ptr,        # (K, N) float32
    C_ptr,        # (M, N) float32 output
    M, N, K,
    stride_a0, stride_a1,
    stride_b0, stride_b1,
    stride_c0, stride_c1,
    bias_ptr,     # (N,) float32
    scale,        # float32
    hmin,         # float32
    hmax,         # float32
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    FP32 variant: loads/compute in float32 (used as fallback).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offsets < M
    n_mask = n_offsets < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        a_ptrs = A_ptr + (m_offsets[:, None] * stride_a0) + (k_offsets[None, :] * stride_a1)
        a_mask = m_mask[:, None] & k_mask[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        b_ptrs = B_ptr + (k_offsets[:, None] * stride_b0) + (n_offsets[None, :] * stride_b1)
        b_mask = k_mask[:, None] & n_mask[None, :]
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

    if bias_ptr is not None:
        bias_vals = tl.load(bias_ptr + n_offsets, mask=n_mask, other=0.0)
        acc = acc + bias_vals[None, :]

    acc = acc * scale
    acc = tl.minimum(tl.maximum(acc, hmin), hmax)

    z = 1.702 * acc
    sig = 1.0 / (1.0 + tl.exp(-z))
    acc = acc * sig

    c_ptrs = C_ptr + (m_offsets[:, None] * stride_c0) + (n_offsets[None, :] * stride_c1)
    store_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(c_ptrs, acc, mask=store_mask)


def triton_fused_linear_activation(A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor,
                                   scale: float, hmin: float, hmax: float, use_fp16: bool = True):
    """
    Wrapper that prepares inputs and launches the appropriate fused Triton kernel.
    A: (M, K) input
    B: (K, N) weight transposed (i.e., original weight.T) in the dtype expected
    bias: (N,)
    Returns float32 output (M, N)
    """
    assert A.is_cuda and B.is_cuda and bias.is_cuda, "All tensors must be on CUDA"
    M, K = A.shape
    Kb, N = B.shape
    assert Kb == K, f"Incompatible K: {Kb} vs {K}"

    # Prefer FP16 path for throughput on Ampere; pad K to multiple of 128 for tensor cores
    if use_fp16:
        A_ = A.contiguous().half()
        B_ = B.contiguous().half()
        # default to original K until the grid(meta) is called and possibly pads to meta["BLOCK_K"]
        K_to_pass = K
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)
        stride_a0, stride_a1 = A_.stride()
        stride_b0, stride_b1 = B_.stride()
        stride_c0, stride_c1 = C.stride()
        bias_contig = bias.contiguous()
        def grid(meta):
            nonlocal A_, B_, K_to_pass, stride_a0, stride_a1, stride_b0, stride_b1, stride_c0, stride_c1
            # Compute the minimal padding required for the kernel's BLOCK_K
            Kpad = ((K + meta["BLOCK_K"] - 1) // meta["BLOCK_K"]) * meta["BLOCK_K"]
            if Kpad != K:
                # allocate padded buffers only when necessary and replace A_/B_
                A_p = torch.empty((M, Kpad), device=A.device, dtype=A_.dtype)
                A_p.zero_()
                A_p[:, :K].copy_(A_)
                B_p = torch.empty((Kpad, N), device=B.device, dtype=B_.dtype)
                B_p.zero_()
                B_p[:K, :].copy_(B_)
                A_, B_ = A_p, B_p
                K_to_pass = Kpad
            # refresh strides in case padding happened
            stride_a0, stride_a1 = A_.stride()
            stride_b0, stride_b1 = B_.stride()
            stride_c0, stride_c1 = C.stride()
            return ( (M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
                     (N + meta["BLOCK_N"] - 1) // meta["BLOCK_N"], )
        _fused_gemm_activation_fp16[grid](
            A_, B_, C,
            M, N, K_to_pass,
            stride_a0, stride_a1,
            stride_b0, stride_b1,
            stride_c0, stride_c1,
            bias_contig,
            float(scale),
            float(hmin),
            float(hmax),
        )
        return C
    else:
        A_ = A.contiguous().to(dtype=torch.float32)
        B_ = B.contiguous().to(dtype=torch.float32)
        # default to original K until the grid(meta) is called and possibly pads to meta["BLOCK_K"]
        K_to_pass = K
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)
        stride_a0, stride_a1 = A_.stride()
        stride_b0, stride_b1 = B_.stride()
        stride_c0, stride_c1 = C.stride()
        bias_contig = bias.contiguous()
        def grid(meta):
            nonlocal A_, B_, K_to_pass, stride_a0, stride_a1, stride_b0, stride_b1, stride_c0, stride_c1
            Kpad = ((K + meta["BLOCK_K"] - 1) // meta["BLOCK_K"]) * meta["BLOCK_K"]
            if Kpad != K:
                A_p = torch.empty((M, Kpad), device=A.device, dtype=A_.dtype)
                A_p.zero_()
                A_p[:, :K].copy_(A_)
                B_p = torch.empty((Kpad, N), device=B.device, dtype=B_.dtype)
                B_p.zero_()
                B_p[:K, :].copy_(B_)
                A_, B_ = A_p, B_p
                K_to_pass = Kpad
            stride_a0, stride_a1 = A_.stride()
            stride_b0, stride_b1 = B_.stride()
            stride_c0, stride_c1 = C.stride()
            return ( (M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
                     (N + meta["BLOCK_N"] - 1) // meta["BLOCK_N"], )
        _fused_gemm_activation_fp32[grid](
            A_, B_, C,
            M, N, K_to_pass,
            stride_a0, stride_a1,
            stride_b0, stride_b1,
            stride_c0, stride_c1,
            bias_contig,
            float(scale),
            float(hmin),
            float(hmax),
        )
        return C


class ModelNew(nn.Module):
    """
    Fused implementation of:
      out = GELU(Hardtanh(Linear(x) * scaling_factor))
    The linear (GEMM), scaling, hardtanh clamp, and GELU activation are fused into a single Triton kernel.
    This implementation caches a half-precision transposed weight to avoid repeated conversions.
    """
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weight and bias similarly to nn.Linear
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float32))

        # Kaiming uniform initialization similar to nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

        # Activation params
        self.scaling_factor = float(scaling_factor)
        self.hardtanh_min = float(hardtanh_min)
        self.hardtanh_max = float(hardtanh_max)

        # Cache the transposed weight as FP16 for fast GPU path and a contiguous bias
        # Registered as buffers so they move with .to(device)
        self.register_buffer('weight_t_half', self.weight.t().contiguous().half())
        self.register_buffer('bias_contig', self.bias.contiguous())

        # Track versions for detecting in-place updates on parameters
        self._weight_version = int(self.weight._version)
        self._bias_version = int(self.bias._version)

    def forward(self, x: torch.Tensor):
        """
        x: (batch_size, in_features)
        returns: (batch_size, out_features) in float32
        """
        # CPU fallback: use PyTorch operations
        if not x.is_cuda:
            out = torch.nn.functional.linear(x, self.weight, self.bias)
            out = out * self.scaling_factor
            out = torch.clamp(out, min=self.hardtanh_min, max=self.hardtanh_max)
            return torch.nn.functional.gelu(out)

        # Refresh cached buffers if weights/bias were changed in-place
        if int(self.weight._version) != int(self._weight_version):
            # Update cached transposed half precision weight
            self.weight_t_half = self.weight.t().contiguous().half()
            self._weight_version = int(self.weight._version)

        if int(self.bias._version) != int(self._bias_version):
            self.bias_contig = self.bias.contiguous()
            self._bias_version = int(self.bias._version)

        # B should be (K, N) where K=in_features, N=out_features
        B = self.weight_t_half  # shape (in_features, out_features) dtype half
        bias = self.bias_contig  # shape (out_features,) dtype float32

        # Launch the fused Triton kernel. Use FP16 path on CUDA for best throughput.
        return triton_fused_linear_activation(x, B, bias,
                                              self.scaling_factor,
                                              self.hardtanh_min,
                                              self.hardtanh_max,
                                              use_fp16=True)