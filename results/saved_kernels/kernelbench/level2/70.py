import torch
import torch.nn as nn
import triton
import triton.language as tl

# Tuned autotune configs for NVIDIA A6000 (Ampere).
# Larger BLOCK_N/BLOCK_M to increase arithmetic intensity for big GEMMs (N=M=8192).
# BLOCK_K chosen to balance WMMA tile shapes and L2 reuse.
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_N": 256, "BLOCK_M": 256, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_N": 256, "BLOCK_M": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_N": 128, "BLOCK_M": 256, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_N": 128, "BLOCK_M": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_N": 64,  "BLOCK_M": 128, "BLOCK_K": 64}, num_warps=4, num_stages=2),
]


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["N", "M", "K"])
@triton.jit
def _fused_gemm_sigmoid_residual_kernel(
    x_ptr,           # pointer to input X (N, K) in fp16
    w_ptr,           # pointer to weight_t (K, M) in fp16 (transposed weight)
    bias_ptr,        # pointer to bias (M,) in fp32
    out_ptr,         # pointer to output (N, M) in fp32
    N, M, K,         # sizes
    stride_xn, stride_xk,
    stride_wm, stride_wk,  # (stride along M, stride along K) for weight_t
    stride_outn, stride_outm,
    scaling,         # python float scalar (not constexpr)
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Kernel computes a BLOCK_N x BLOCK_M tile of:
      Y = X @ W^T + bias  (X: N x K, W: M x K  -> weight_t: K x M)
      Out = Y + scaling * sigmoid(Y)

    Inputs X and weight_t are expected in fp16 to utilize Tensor Cores.
    Accumulation and activation performed in fp32 for numerical stability.
    """
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    n_start = pid_n * BLOCK_N
    m_start = pid_m * BLOCK_M

    offs_n = n_start + tl.arange(0, BLOCK_N)
    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    # accumulator in fp32
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k = k_start + offs_k  # shape (BLOCK_K,)

        # Load X block: shape (BLOCK_N, BLOCK_K) in fp16, mask on bounds
        x_addr = x_ptr + offs_n[:, None] * stride_xn + k[None, :] * stride_xk
        mask_x = (offs_n[:, None] < N) & (k[None, :] < K)
        x_block = tl.load(x_addr, mask=mask_x, other=0.0)  # fp16

        # Load W block from transposed weight_t (K, M): shape (BLOCK_K, BLOCK_M) in fp16
        w_addr = w_ptr + k[:, None] * stride_wk + offs_m[None, :] * stride_wm
        mask_w = (k[:, None] < K) & (offs_m[None, :] < M)
        w_block = tl.load(w_addr, mask=mask_w, other=0.0)  # fp16

        # Accumulate: tl.dot will accumulate to fp32 when inputs are fp16
        acc += tl.dot(x_block, w_block)

    # Add bias (fp32) - broadcast across N rows
    bias_addr = bias_ptr + offs_m
    mask_b = offs_m < M
    bias_vals = tl.load(bias_addr, mask=mask_b, other=0.0)  # fp32
    acc = acc + bias_vals[None, :]

    # Compute sigmoid in fp32: sigmoid(x) = 1 / (1 + exp(-x))
    neg = -acc
    exp_neg = tl.exp(neg)
    sig = 1.0 / (1.0 + exp_neg)

    # Final output: y + scaling * sigmoid(y)
    out_tile = acc + scaling * sig

    # Store to output (fp32)
    out_addr = out_ptr + offs_n[:, None] * stride_outn + offs_m[None, :] * stride_outm
    mask_out = (offs_n[:, None] < N) & (offs_m[None, :] < M)
    tl.store(out_addr, out_tile, mask=mask_out)


def _triton_fused_forward(x: torch.Tensor, weight_t: torch.Tensor, bias: torch.Tensor, scaling: float):
    """
    Wrapper to launch the Triton fused kernel.
    x: (N, K) fp16 tensor on CUDA
    weight_t: (K, M) fp16 transposed weight on CUDA
    bias: (M,) fp32 tensor on CUDA
    returns out: (N, M) fp32 tensor on CUDA
    """
    assert x.is_cuda and weight_t.is_cuda and bias.is_cuda, "All tensors must be on CUDA"
    x = x.contiguous()
    weight_t = weight_t.contiguous()
    bias = bias.contiguous()

    N, K = x.shape
    K_w, M = weight_t.shape
    assert K == K_w, "K dimension mismatch between x and weight_t"

    out = torch.empty((N, M), device=x.device, dtype=torch.float32)

    # strides (in elements)
    stride_xn, stride_xk = x.stride(0), x.stride(1)
    # weight_t has shape (K, M) so stride_wk is stride along K (dim0), stride_wm along M (dim1)
    stride_wk, stride_wm = weight_t.stride(0), weight_t.stride(1)
    stride_outn, stride_outm = out.stride(0), out.stride(1)

    # grid based on BLOCK sizes selected by autotune
    def grid(meta):
        bn = (N + meta["BLOCK_N"] - 1) // meta["BLOCK_N"]
        bm = (M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"]
        return (bn, bm)

    # Note: kernel expects args (stride_wm, stride_wk) ordering, so pass swapped accordingly.
    _fused_gemm_sigmoid_residual_kernel[grid](
        x, weight_t, bias, out,
        N, M, K,
        stride_xn, stride_xk,
        stride_wm, stride_wk,  # pass in order (stride_wm, stride_wk)
        stride_outn, stride_outm,
        float(scaling),
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model using a fused Triton kernel for:
      Linear (GEMM) -> Sigmoid -> Scaling -> ResidualAdd

    This model keeps parameters similar to nn.Linear (weight and bias), but
    uses a mixed-precision Triton kernel for the heavy GEMM and fused ops.
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        # weight: (out_features, in_features) to mimic nn.Linear
        self.weight = nn.Parameter(torch.empty(hidden_size, input_size))
        self.bias = nn.Parameter(torch.empty(hidden_size))
        self.scaling_factor = float(scaling_factor)

        # initialize parameters similar to nn.Linear
        bound = 1.0 / input_size ** 0.5
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

        # cached transposed weight (fp16) to avoid re-transposing each forward
        self._weight_t = None
        self._weight_t_ptr = None
        self._weight_t_h = None
        self._weight_t_h_ptr = None

    def forward(self, x: torch.Tensor):
        """
        Forward using fused Triton kernel.

        x: (batch_size, input_size) float32 or float16 tensor.
        Returns:
          out: (batch_size, hidden_size) float32 tensor
        """
        # Fallback to CPU / pure PyTorch implementation if input not on CUDA
        if not x.is_cuda:
            # Use same ops as original: linear -> sigmoid -> scale -> residual
            y = torch.nn.functional.linear(x, self.weight, self.bias)
            return y + self.scaling_factor * torch.sigmoid(y)

        # Ensure contiguous input on CUDA
        x = x.contiguous()

        # Prepare and cache transposed weight in fp16 for the kernel
        weight = self.weight
        weight_ptr = weight.data_ptr()
        need_recompute = (getattr(self, "_weight_t_ptr", None) != weight_ptr) or (getattr(self, "_weight_t", None) is None)
        if need_recompute:
            # create contiguous transposed weight (K, M) where K=input_size, M=hidden_size
            weight_t = weight.t().contiguous()  # shape (K, M) in fp32
            self._weight_t = weight_t
            self._weight_t_ptr = weight_ptr
            # fp16 cached copy for kernel
            self._weight_t_h = weight_t.half().contiguous()
            self._weight_t_h_ptr = weight_ptr
        else:
            # ensure fp16 view exists
            if getattr(self, "_weight_t_h", None) is None:
                self._weight_t_h = self._weight_t.half().contiguous()
                self._weight_t_h_ptr = weight_ptr

        # Mixed precision: cast input to fp16 and use cached fp16 transposed weight.
        x_h = x.half()
        weight_t_h = self._weight_t_h
        bias = self.bias.contiguous()

        return _triton_fused_forward(x_h, weight_t_h, bias, self.scaling_factor)


# Keep helper values similar to the original module
batch_size = 1024
input_size = 8192
hidden_size = 8192
scaling_factor = 2.0


def get_inputs():
    return [torch.rand(batch_size, input_size).cuda()]


def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]