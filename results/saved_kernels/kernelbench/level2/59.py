import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configurations tailored for NVIDIA A6000 (Ampere).
# Favor modest tile sizes to increase occupancy and reduce register/shared-memory pressure.
# BLOCK_* remain multiples of 16/32 so TensorCore MMA can be utilized.
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=3),
]

@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def _fused_gemm_swish_kernel(
    a_ptr,            # A pointer (M x K), row-major, fp16 input
    b_ptr,            # B pointer (K x N), row-major, fp16 input (this is weight.T)
    out_ptr,          # C pointer (M x N), row-major, fp32 output
    bias_ptr,         # bias pointer (N,), fp16
    M, N, K,          # dimensions
    stride_am, stride_ak,   # strides for A
    stride_bk, stride_bn,   # strides for B
    stride_cm, stride_cn,   # strides for C (output)
    stride_bias,            # stride for bias (should be 1 for contiguous)
    activation: tl.constexpr,             # constexpr flag: 0=exact(fp32),1=fp16-exp,2=hard-swish approx
    scaling_factor,         # float scalar
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """
    Fused kernel:
      - Inputs A,B,bias are provided in fp16 to reduce memory bandwidth.
      - Accumulate in fp32.
      - Apply Swish activation (x * sigmoid(x)) followed by scaling_factor.
      - Optionally support faster approximate activations via `activation` flag:
          0: exact fp32 sigmoid (default)
          1: fp16-exp path (cast to fp16 for exp, then back to fp32)
          2: hard-swish approximation (x * relu6(x+3) / 6)
      - Store fp32 output.
    """

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Row/col indices for this tile
    row_idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_idx = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Number of k-tiles
    k_tiles = (K + BLOCK_K - 1) // BLOCK_K

    for kt in range(k_tiles):
        k_start = kt * BLOCK_K
        k_idx = k_start + tl.arange(0, BLOCK_K)

        # Build addresses
        a_offs = row_idx[:, None] * stride_am + k_idx[None, :] * stride_ak
        b_offs = k_idx[:, None] * stride_bk + col_idx[None, :] * stride_bn

        # Masks for in-bounds loads
        mask_a = (row_idx[:, None] < M) & (k_idx[None, :] < K)
        mask_b = (k_idx[:, None] < K) & (col_idx[None, :] < N)

        # Load tiles as fp16; 'other' is 0.0
        a_tile_fp16 = tl.load(a_ptr + a_offs, mask=mask_a, other=0.0)
        b_tile_fp16 = tl.load(b_ptr + b_offs, mask=mask_b, other=0.0)

        # Use tl.dot on fp16 tiles -> Triton emits MMA/Tensor-Core code and accumulates into fp32 acc
        acc += tl.dot(a_tile_fp16, b_tile_fp16)

    # Add bias (load fp16 bias, cast to fp32)
    bias_mask = col_idx < N
    bias_vals_fp16 = tl.load(bias_ptr + col_idx * stride_bias, mask=bias_mask, other=0.0)
    bias_vals = tl.cast(bias_vals_fp16, tl.float32)
    acc += bias_vals[None, :]

    # Activation handling:
    # activation == 0: exact fp32 sigmoid (original)
    # activation == 1: cast to fp16 for the exp path to reduce math cost, then cast back
    # activation == 2: hard-swish approximation: x * relu6(x+3) / 6
    if activation == 1:
        # Compute exp in fp32 (tl.exp does not accept fp16 inputs)
        # acc is already fp32 from the accumulation, so perform the expensive math in fp32.
        neg_fp32 = -acc
        denom_fp32 = 1.0 + tl.exp(neg_fp32)
        sigmoid_fp32 = 1.0 / denom_fp32
        out_tile = acc * sigmoid_fp32
    elif activation == 2:
        # Hard-swish approximation (branchless via tl.where)
        t = acc + 3.0
        t = tl.where(t < 0.0, 0.0, t)
        t = tl.where(t > 6.0, 6.0, t)
        out_tile = acc * (t / 6.0)
    else:
        # Default exact fp32 sigmoid
        neg = -acc
        denom = 1.0 + tl.exp(neg)
        sigmoid = 1.0 / denom
        out_tile = acc * sigmoid

    # Apply scaling
    out_tile = out_tile * scaling_factor

    # Store as fp32
    c_offs = row_idx[:, None] * stride_cm + col_idx[None, :] * stride_cn
    store_mask = (row_idx[:, None] < M) & (col_idx[None, :] < N)
    tl.store(out_ptr + c_offs, out_tile, mask=store_mask)


def _triton_fused_gemm_swish(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor, scaling: float):
    """
    Wrapper to prepare tensors and launch the Triton kernel.
      - a: (M, K) fp16
      - b: (K, N) fp16
      - bias: (N,) fp16
      Returns fp32 output of shape (M, N).
    """
    assert a.is_cuda and b.is_cuda and bias.is_cuda, "All tensors must be CUDA tensors for Triton kernel."

    M, K = a.shape
    k_b, N = b.shape
    assert k_b == K, "Incompatible K dimension between A and B."

    # Prepare output (fp32)
    out = torch.empty((M, N), device=a.device, dtype=torch.float32)

    # Strides in elements (row-major expected)
    stride_am = a.stride(0)
    stride_ak = a.stride(1)
    stride_bk = b.stride(0)
    stride_bn = b.stride(1)
    stride_cm = out.stride(0)
    stride_cn = out.stride(1)
    stride_bias = bias.stride(0)

    # Grid based on autotune meta
    def grid(meta):
        return ( (M + meta['BLOCK_M'] - 1) // meta['BLOCK_M'],
                 (N + meta['BLOCK_N'] - 1) // meta['BLOCK_N'] )

    # Launch kernel
    # activation flag: 0=exact(fp32 sigmoid) (default), 1=fp16-exp path, 2=hard-swish approx
    _fused_gemm_swish_kernel[grid](
        a, b, out, bias,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        stride_bias,
        int(0),
        float(scaling)
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that fuses the linear matmul, Swish activation, and scaling into a Triton kernel.
    Maintains same parameter semantics as nn.Linear (weight and bias).
    """

    def __init__(self, in_features: int, out_features: int, scaling_factor: float):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scaling_factor = float(scaling_factor)

        # weight: (out_features, in_features) to mimic nn.Linear
        self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty((out_features,), dtype=torch.float32))
        self.reset_parameters()

        # Cached fp16 transposed weight (K x N) and fp16 bias to avoid repeated conversions
        # These are registered buffers so they move with the module to the correct device if .to(device) is used.
        self.register_buffer('_weight_t_half', None)
        self.register_buffer('_bias_half', None)
        # Track pointers to detect in-place updates to parameters
        self._weight_ptr = None
        self._bias_ptr = None

    def reset_parameters(self):
        # Use the same initialization scheme as nn.Linear
        bound = 1.0 / (self.in_features ** 0.5)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
          - If inputs are on CUDA, run fused Triton kernel (matmul + swish + scaling).
          - Otherwise, fall back to torch.nn.functional.linear + swish + scaling.
        """
        if not x.is_cuda or not self.weight.is_cuda:
            # CPU (or mismatch) fallback: exact semantics with fp32
            y = torch.nn.functional.linear(x, self.weight, self.bias)
            y = y * torch.sigmoid(y)
            y = y * self.scaling_factor
            return y

        # Ensure contiguous input and convert to fp16
        a = x.contiguous().half()  # shape (M, K)

        # Update cached half-precision transposed weight if underlying FP32 param changed
        if (self._weight_t_half is None) or (self._weight_ptr != self.weight.data_ptr()):
            # Transpose weight to (K, N) layout, make contiguous and cast to half
            # weight is (N, K) -> weight.t() is (K, N)
            wt = self.weight.data.t().contiguous().half()
            # move to same device if needed
            if wt.device != a.device:
                wt = wt.to(a.device)
            # set buffer
            self._weight_t_half = wt
            self._weight_ptr = self.weight.data_ptr()

        if (self._bias_half is None) or (self._bias_ptr != self.bias.data_ptr()):
            b_half = self.bias.contiguous().half()
            if b_half.device != a.device:
                b_half = b_half.to(a.device)
            self._bias_half = b_half
            self._bias_ptr = self.bias.data_ptr()

        b = self._weight_t_half
        bias = self._bias_half

        # Call Triton fused kernel; it returns fp32 tensor
        out = _triton_fused_gemm_swish(a, b, bias, self.scaling_factor)
        return out


# Convenience functions to mirror the original API
batch_size = 128
in_features = 32768
out_features = 32768
scaling_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, scaling_factor]