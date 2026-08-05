import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configs tuned for NVIDIA A6000 / Ampere to favor larger tiles and Tensor Cores.
# BLOCK_K are multiples of 8; BLOCK_M/BLOCK_N are multiples of 32 to align with warps/lanes.
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 512, "BLOCK_N": 512, "BLOCK_K": 128}, num_warps=16, num_stages=4),
    triton.Config({"BLOCK_M": 512, "BLOCK_N": 256, "BLOCK_K": 128}, num_warps=16, num_stages=4),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 128}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=2),
]


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _matmul_fused_kernel(
    A_ptr,  # A: (M, K) row-major
    B_ptr,  # B: (K, N) row-major (i.e., weight.t().contiguous() or padded half transposed)
    C_ptr,  # C: (M, N) row-major (fp32)
    bias_ptr,  # bias: (N,) or empty
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    scale_plus_one,  # precomputed (scale + 1.0)
    has_bias,        # 0/1 runtime flag
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offs = tl.arange(0, BLOCK_K)

    # accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks of BLOCK_K
    for k_start in range(0, K, BLOCK_K):
        k_iter = k_start + k_offs  # shape (BLOCK_K,)

        # A tile pointers (BLOCK_M, BLOCK_K)
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + k_iter[None, :] * stride_ak)
        mask_a = (offs_m[:, None] < M) & (k_iter[None, :] < K)
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)

        # B tile pointers (BLOCK_K, BLOCK_N) since B is (K, N)
        b_ptrs = B_ptr + (k_iter[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        mask_b = (k_iter[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)

        # accumulate (handles half inputs by accumulating to fp32 if types are half)
        acc += tl.dot(a, b)

    # add bias if present
    if has_bias != 0:
        n_mask = offs_n < N
        bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
        acc = acc + bias[None, :]

    # fuse scaling + residual by multiplying accumulator by (scale + 1.0)
    acc *= tl.cast(scale_plus_one, tl.float32)

    # store output
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask_c)


def triton_gemm_fused(
    a: torch.Tensor,
    b_t: torch.Tensor,
    bias: torch.Tensor = None,
    scale_plus_one: float = 1.0,
):
    """
    Wrapper to call Triton matmul fused kernel.
    - a: (M, K), contiguous
    - b_t: (K, N), contiguous (may be half or float)
    Returns float32 output (M, N).
    """
    assert a.is_cuda and b_t.is_cuda, "Tensors must be on CUDA."
    assert a.dtype in (torch.float32, torch.float16)
    assert b_t.dtype in (torch.float32, torch.float16)

    A = a.contiguous()
    B = b_t.contiguous()
    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb, "K mismatch between A and B"

    # prepare output as float32
    out = torch.empty((M, N), device=A.device, dtype=torch.float32)

    if bias is None:
        bias_t = torch.empty((0,), device=A.device, dtype=torch.float32)
        has_bias = 0
    else:
        bias_t = bias.contiguous().to(torch.float32)
        has_bias = 1

    stride_am = A.stride(0)
    stride_ak = A.stride(1)
    stride_bk = B.stride(0)
    stride_bn = B.stride(1)
    stride_cm = out.stride(0)
    stride_cn = out.stride(1)

    def grid(meta):
        bm = meta["BLOCK_M"]
        bn = meta["BLOCK_N"]
        return ((M + bm - 1) // bm, (N + bn - 1) // bn)

    # call kernel; pass precomputed scale_plus_one to reduce in-kernel ops
    _matmul_fused_kernel[grid](
        A, B, out, bias_t,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        float(scale_plus_one),
        int(has_bias),
    )
    return out


class TritonLinearMixedFunction(torch.autograd.Function):
    """
    Autograd function that runs a Triton fused GEMM in forward under no_grad to implement the
    detached-residual semantics, and implements backward such that only the scaled path contributes.
    Forward signature:
      input, weight, bias, scaling_factor, weight_t_float, weight_t_half, use_fp16
    We return gradients for input, weight, bias and None for other non-leaf args.
    """

    @staticmethod
    def forward(ctx, input, weight, bias, scaling_factor, weight_t_float, weight_t_half, use_fp16):
        # Move tensors to weight device if needed
        device = weight.device
        if not input.is_cuda and device.type == "cuda":
            input = input.cuda(device)
        if bias is not None and (not bias.is_cuda) and device.type == "cuda":
            bias = bias.cuda(device)

        # Choose mixed-precision path when requested and available
        chosen_b = weight_t_float
        chosen_a = input
        use_half = bool(use_fp16) and (weight_t_half is not None)
        if use_half:
            # use half buffers for b to leverage Tensor Cores; avoid per-call padding/copies for A.
            # We rely on the Triton kernel's masked loads to handle K tails. This avoids a large allocation
            # on every forward. Convert input to half and make contiguous only.
            chosen_b = weight_t_half
            chosen_a = input.half().contiguous()
            # Do not pad chosen_a here; the kernel handles K tails with masking.
        else:
            # float path; ensure a,b are float32
            chosen_a = input.contiguous()
            chosen_b = weight_t_float.contiguous()

        # precompute scale + 1.0 for in-kernel fused multiply
        scale_plus_one = float(scaling_factor) + 1.0

        # Run triton kernel under no_grad to emulate detach() residual
        with torch.no_grad():
            out = triton_gemm_fused(chosen_a, chosen_b, bias=bias, scale_plus_one=scale_plus_one)

        # Save tensors for backward: original input and original non-transposed weight (float32)
        ctx.save_for_backward(input, weight)
        ctx.bias = bias
        ctx.scaling_factor = float(scaling_factor)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        bias = ctx.bias
        s = ctx.scaling_factor

        # Only the scaled path contributes to gradient (detached residual)
        grad_acc = grad_output * s

        needs = ctx.needs_input_grad
        # forward inputs were:
        # (input, weight, bias, scaling_factor, weight_t_float, weight_t_half, use_fp16)
        grad_input = grad_weight = grad_bias = None
        # non-tensor/aux args will receive None gradients

        if needs[0]:
            # grad_input = grad_acc @ weight  (weight: N x K)
            grad_input = grad_acc.mm(weight)
        if needs[1]:
            # grad_weight = grad_acc.T @ input  -> (N x M) @ (M x K) => (N x K)
            grad_weight = grad_acc.t().mm(input)
        if needs[2] and (bias is not None):
            grad_bias = grad_acc.sum(0)

        # Return gradients for all forward inputs in order:
        # (input, weight, bias, scaling_factor, weight_t_float, weight_t_half, use_fp16)
        return grad_input, grad_weight, grad_bias, None, None, None, None


class ModelNew(nn.Module):
    """
    Optimized Model using Triton GEMM with fused scaling+residual. This implementation:
      - caches transposed weight buffers (float32 and padded half) to avoid repeated transposes.
      - uses a mixed-precision (fp16) path when available to leverage Tensor Cores on Ampere.
      - fuses the (scale + residual) operation into the GEMM kernel by computing acc * (scale + 1).
      - preserves detached-residual semantics by running the forward GEMM under no_grad and implementing
        backward such that only the scaled path contributes to gradients.
    """

    def __init__(self, in_features, out_features, scaling_factor, use_fp16: bool = True):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = float(scaling_factor)
        self.use_fp16 = bool(use_fp16)

        # cached transposed buffers (not Parameters)
        self._weight_t = None        # float32 transposed (K, N)
        self._weight_t_half = None   # half transposed, padded on K to multiple of 8
        self._cached_weight_ptr = None

    def _ensure_weight_t_cached(self):
        w = self.matmul.weight
        ptr = w.data_ptr()
        if (self._weight_t is None) or (self._cached_weight_ptr != ptr):
            # float32 transposed contiguous
            wt = w.t().contiguous()
            self._weight_t = wt

            # create half transposed and pad K to multiple of 8 for Tensor Core efficiency
            try:
                wth = wt.half().contiguous()
                K, N = wth.shape
                pad_k = (8 - (K % 8)) % 8
                if pad_k != 0:
                    wth_padded = torch.zeros((K + pad_k, N), device=wth.device, dtype=wth.dtype)
                    wth_padded[:K, :] = wth
                    self._weight_t_half = wth_padded.contiguous()
                else:
                    self._weight_t_half = wth
            except Exception:
                # If half conversion fails for any reason, drop the half cache
                self._weight_t_half = None

            self._cached_weight_ptr = ptr

        return self._weight_t, self._weight_t_half

    def forward(self, x):
        # Move inputs to device of weights if needed
        device = self.matmul.weight.device
        if not x.is_cuda and device.type == "cuda":
            x = x.cuda(device)
        if self.matmul.bias is not None and (not self.matmul.bias.is_cuda) and device.type == "cuda":
            self.matmul.bias = self.matmul.bias.cuda(device)

        # Ensure cached transposed buffers are up-to-date
        wt_float, wt_half = self._ensure_weight_t_cached()

        # Call custom autograd function. We pass both cached transposed buffers and the original weight
        # so forward can pick the best path and backward has access to the original weight for gradients.
        out = TritonLinearMixedFunction.apply(
            x,                       # input (M, K)
            self.matmul.weight,      # original weight (N, K) float32
            self.matmul.bias,        # bias (N,) or None
            self.scaling_factor,     # scalar
            wt_float,                # transposed weight float32 (K, N)
            wt_half,                 # optional padded transposed half (K_padded, N)
            self.use_fp16,           # whether to use fp16 path
        )
        return out