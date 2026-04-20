import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune search space targeted for Ampere A6000.
# Try a small set of configurations for BLOCK_M, BLOCK_K, BLOCK_N and num_warps that work well on Ampere.
AUTOTUNE_CONFIGS = []
for BM in (32, 64):
    for BK in (32, 64, 128):
        for BN in (128, 256, 512):
            for nw in (4, 8):
                AUTOTUNE_CONFIGS.append(triton.Config({"BLOCK_M": BM, "BLOCK_N": BN, "BLOCK_K": BK}, num_warps=nw, num_stages=2))

# Reduced tile sizes to avoid shared-memory exhaustion on A6000.
# Also provide a runtime flag to the kernel to optionally write fp16 outputs (for intermediate layers)
# while still performing fp32 accumulation + bias+ReLU for numerical safety.
@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['M', 'N', 'K', 'lda', 'ldb', 'ldc'])
@triton.jit
def _matmul_bias_relu_kernel(
    a_ptr,        # pointer to A (M x K) in fp16
    b_ptr,        # pointer to W_T (K x N) in fp16 -- weights expected transposed (K, N)
    c_ptr,        # pointer to C (M x N) output (fp32 or fp16 depending on OUT_FP16)
    M, N, K,      # sizes
    lda, ldb, ldc,# strides (in elements)
    bias_ptr,     # pointer to bias vector of size N (fp32)
    RELU,         # runtime flag (0 or 1) to apply ReLU
    OUT_FP16,     # runtime flag (0 or 1) to store output as fp16
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """
    Compute a tile of C = A @ W where:
      A: (M, K) fp16
      Wt: (K, N) fp16
    Load fp16 data and perform the matmul on fp16 tiles so Triton/hardware can use Tensor Cores.
    The accumulator `acc` is fp32 and receives the (mixed-precision) result; bias and optional ReLU
    are applied in fp32. If OUT_FP16 != 0 the accumulated fp32 tile is cast to fp16 and stored;
    otherwise fp32 is stored.
    """

    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)

    row_start = row_idx * BLOCK_M
    col_start = col_idx * BLOCK_N

    offs_m = row_start + tl.arange(0, BLOCK_M)    # (BLOCK_M,)
    offs_n = col_start + tl.arange(0, BLOCK_N)    # (BLOCK_N,)

    # accumulator for the tile (fp32)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Precompute 1D bounds masks once per tile to reduce boolean ops inside the K-loop.
    row_mask = offs_m < M    # (BLOCK_M,)
    col_mask = offs_n < N    # (BLOCK_N,)

    # Iterate K in chunks of BLOCK_K and perform blocked 2D loads + matrix multiply (tl.dot).
    k = 0
    while k < K:
        offs_k = k + tl.arange(0, BLOCK_K)   # (BLOCK_K,)
        k_mask = offs_k < K

        # Build 2D masks by broadcasting the 1D masks (cheaper than recomputing full 2D comparisons each iteration).
        mask_a = row_mask[:, None] & k_mask[None, :]   # (BLOCK_M, BLOCK_K)
        mask_b = k_mask[:, None] & col_mask[None, :]   # (BLOCK_K, BLOCK_N)

        # pointers for 2D tiles
        a_ptrs = a_ptr + (offs_m[:, None] * lda + offs_k[None, :])  # (BLOCK_M, BLOCK_K)
        b_ptrs = b_ptr + (offs_k[:, None] * ldb + offs_n[None, :])  # (BLOCK_K, BLOCK_N)

        # load tiles (masked) and keep as fp16 for the dot
        a_tile_fp16 = tl.load(a_ptrs, mask=mask_a, other=0.0)   # (BLOCK_M, BLOCK_K) fp16
        b_tile_fp16 = tl.load(b_ptrs, mask=mask_b, other=0.0)   # (BLOCK_K, BLOCK_N) fp16

        # accumulate via tile matmul using fp16 tiles; acc is fp32 so accumulation is in fp32 (mixed-precision)
        acc += tl.dot(a_tile_fp16, b_tile_fp16)

        k += BLOCK_K

    # add bias (fp32) (broadcast across M)
    bias_mask = col_mask
    bias_vals = tl.load(bias_ptr + offs_n, mask=bias_mask, other=0.0)  # (BLOCK_N,) fp32
    acc = acc + bias_vals[None, :]

    # optional ReLU fusion (fp32)
    if RELU != 0:
        acc = tl.where(acc > 0.0, acc, 0.0)

    # store with bounds checking into either fp32 or fp16 output
    store_mask = row_mask[:, None] & col_mask[None, :]
    c_ptrs = c_ptr + (offs_m[:, None] * ldc + offs_n[None, :])

    if OUT_FP16 != 0:
        # cast accumulator to fp16 and store
        out_tile_fp16 = tl.cast(acc, tl.float16)
        tl.store(c_ptrs, out_tile_fp16, mask=store_mask)
    else:
        tl.store(c_ptrs, acc, mask=store_mask)


def triton_matmul_bias_relu(A: torch.Tensor, Wt: torch.Tensor, bias: torch.Tensor, relu: bool = False, return_fp16: bool = False, out: torch.Tensor = None):
    """
    Compute Y = A @ W.T + bias with optional ReLU using Triton kernel.

    A: (M, K) host tensor, can be torch.float32 or torch.float16. If float32, it will be cast to fp16 (host->device).
    Wt: (K, N) host tensor, expected transposed layout (K,N) and ideally fp16 (ModelNew caches fp16 transposed weights).
    bias: (N,) host fp32 tensor (kernel expects fp32 bias).
    return_fp16: if True, the kernel will store fp16 output and this function returns a torch.half tensor.
                 otherwise returns a torch.float32 output tensor.
    out: optional preallocated output tensor. If provided, its device/dtype/shape are validated and the kernel writes into it.
         If out is non-contiguous, a contiguous temporary will be used and copied back into `out` after the kernel.
    """
    assert A.is_cuda and Wt.is_cuda and bias.is_cuda, "All tensors must be on CUDA."
    assert bias.dtype == torch.float32, "Bias must be fp32."

    # create contiguous views (and cast to fp16 only if needed)
    A_ = A.contiguous()
    Wt_ = Wt.contiguous()
    bias_ = bias.contiguous()

    # prepare fp16 inputs for the kernel; avoid repeated casts if already fp16
    if A_.dtype == torch.float16:
        A_fp16 = A_
    else:
        A_fp16 = A_.half().contiguous()

    if Wt_.dtype == torch.float16:
        Wt_fp16 = Wt_
    else:
        Wt_fp16 = Wt_.half().contiguous()

    # host-side shapes (use fp16 views shapes)
    M, K = A_fp16.shape
    assert Wt_fp16.shape[0] == K, "Wt should have shape (K, N) matching A's K"
    N = Wt_fp16.shape[1]

    # determine expected dtype for output
    expected_dtype = torch.half if return_fp16 else torch.float32

    # Handle optional output buffer reuse
    need_copy_back = False
    if out is None:
        C = torch.empty((M, N), device=A_fp16.device, dtype=expected_dtype).contiguous()
    else:
        assert out.device == A_fp16.device, "Output buffer must be on the same device as inputs."
        assert out.shape[0] == M and out.shape[1] >= N, "Output buffer must have shape (M, >=N)."
        if out.dtype != expected_dtype:
            raise AssertionError("Provided output tensor has wrong dtype.")
        if out.is_contiguous():
            C = out
        else:
            # create a contiguous temporary and copy back after kernel
            C = out.contiguous()
            need_copy_back = True

    def grid(meta):
        blocks_m = (M + meta['BLOCK_M'] - 1) // meta['BLOCK_M']
        blocks_n = (N + meta['BLOCK_N'] - 1) // meta['BLOCK_N']
        return (blocks_m, blocks_n)

    _matmul_bias_relu_kernel[grid](
        A_fp16, Wt_fp16, C,
        M, N, K,
        A_fp16.stride(0), Wt_fp16.stride(0), C.stride(0),
        bias_,
        int(relu),
        int(return_fp16)
    )

    if need_copy_back:
        out.copy_(C)
        return out
    return C


def triton_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None, relu: bool = False):
    """
    Fused linear using Triton matmul kernel with fused bias and optional ReLU.
    weight: (out_features, in_features)  -- PyTorch layout (N, K)
    x: (batch, in_features)
    bias: (out_features,)
    """
    assert bias is not None, "This fused kernel expects a bias tensor."
    # transpose weight to (K, N) for coalesced loads in the kernel
    Wt = weight.t().contiguous()
    return triton_matmul_bias_relu(x, Wt, bias, relu=relu)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        """
        Same MLP architecture but uses Triton-accelerated fused Linear (GEMM + bias + optional ReLU)
        for the forward pass. Standard nn.Linear modules are retained to store parameters and enable autograd.

        This version caches transposed fp16 weights (Wt_fp16 = weight.t().half().contiguous()) to avoid repeated
        host-side casts for very large weight matrices. During forward, activations are cast to fp16 once and then
        the Triton kernel is called with fp16 activations and fp16 transposed weights.
        Hidden layers request fp16 outputs from the kernel (to reduce write bandwidth and avoid host-side casts),
        while the final layer returns fp32.
        """
        super(ModelNew, self).__init__()
        assert len(hidden_layer_sizes) >= 1, "Expect at least one hidden layer."

        self.fc_layers = nn.ModuleList()
        current = input_size
        for h in hidden_layer_sizes:
            # keep bias=True as original model
            self.fc_layers.append(nn.Linear(current, h, bias=True))
            current = h
        self.out_layer = nn.Linear(current, output_size, bias=True)

        # lazy cache for transposed weights (Wt_fp16 = weight.t().half().contiguous())
        # We also keep the data_ptr() of the original parameter so we only refresh the cache if the parameter buffer changed.
        self._wt_cache = [None] * (len(self.fc_layers) + 1)
        self._wt_cache_ptrs = [0] * (len(self.fc_layers) + 1)

    def forward(self, x):
        # Cast input to fp16 once (on device). Use this fp16 activation as input to kernels.
        batch = x.shape[0]
        device = x.device
        out_fp16 = x.half().contiguous()

        # Prepare reusable ping-pong buffers sized to the largest hidden width to avoid per-layer allocations.
        if not hasattr(self, "_out_bufs"):
            self._out_bufs = [None, None]
            self._out_bufs_sizes = (0, 0)  # (batch, width)

        # compute maximum hidden width among hidden layers (0 if no hidden layers)
        max_hidden = 0
        for fc in self.fc_layers:
            max_hidden = max(max_hidden, fc.out_features)

        # ensure buffers exist and are large enough
        if max_hidden > 0:
            need_new = False
            cur_buf = self._out_bufs[0]
            if (cur_buf is None or cur_buf.shape[0] != batch or cur_buf.shape[1] < max_hidden or cur_buf.device != device):
                need_new = True
            if need_new:
                # allocate two contiguous fp16 buffers sized (batch, max_hidden)
                self._out_bufs[0] = torch.empty((batch, max_hidden), device=device, dtype=torch.half).contiguous()
                self._out_bufs[1] = torch.empty((batch, max_hidden), device=device, dtype=torch.half).contiguous()

        # hidden layers with fused linear + ReLU, writing into the ping-pong buffers
        ping = 0
        for idx, fc in enumerate(self.fc_layers):
            # check param data pointer to avoid unnecessary transposes/casts
            param_ptr = int(fc.weight.data_ptr())
            wt_fp16 = self._wt_cache[idx]
            expected_shape = (fc.weight.shape[1], fc.weight.shape[0])
            if (wt_fp16 is None or
                wt_fp16.device != fc.weight.device or
                wt_fp16.shape != expected_shape or
                self._wt_cache_ptrs[idx] != param_ptr):
                # create contiguous transposed half-precision weight on the correct device and update pointer stamp
                self._wt_cache[idx] = fc.weight.t().half().contiguous()
                self._wt_cache_ptrs[idx] = param_ptr
                wt_fp16 = self._wt_cache[idx]

            # bias should be fp32 for the kernel
            bias = fc.bias.contiguous()

            # select an output buffer (full width = max_hidden). Kernel will only write up to fc.out_features columns.
            if max_hidden > 0:
                out_buf = self._out_bufs[ping]
            else:
                out_buf = None

            # call kernel with fp16 activations and fp16 weight; request fp16 output and reuse out_buf when available
            # The kernel writes only the first fc.out_features columns; the buffer is contiguous and can be reused.
            out_written = triton_matmul_bias_relu(out_fp16, wt_fp16, bias, relu=True, return_fp16=True, out=out_buf)

            # After this call, set out_fp16 to the slice containing the current layer's output (contiguous prefix)
            if max_hidden > 0:
                out_fp16 = out_buf[:, :fc.out_features]
            else:
                out_fp16 = out_written

            # ping-pong
            ping ^= 1

        # final linear (no ReLU) -> return fp32
        idx_out = len(self.fc_layers)
        param_ptr_out = int(self.out_layer.weight.data_ptr())
        wt_out_fp16 = self._wt_cache[idx_out]
        expected_shape_out = (self.out_layer.weight.shape[1], self.out_layer.weight.shape[0])
        if (wt_out_fp16 is None or
            wt_out_fp16.device != self.out_layer.weight.device or
            wt_out_fp16.shape != expected_shape_out or
            self._wt_cache_ptrs[idx_out] != param_ptr_out):
            self._wt_cache[idx_out] = self.out_layer.weight.t().half().contiguous()
            self._wt_cache_ptrs[idx_out] = param_ptr_out
            wt_out_fp16 = self._wt_cache[idx_out]

        bias_out = self.out_layer.bias.contiguous()
        # final output is fp32; allocate a fp32 output (could be cached similarly if desired)
        out_final_fp32 = triton_matmul_bias_relu(out_fp16, wt_out_fp16, bias_out, relu=False, return_fp16=False)
        return out_final_fp32