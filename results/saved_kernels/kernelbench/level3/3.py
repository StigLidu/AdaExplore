import torch
import torch.nn as nn
import triton
import triton.language as tl

# Aggressive autotune configs tailored for NVIDIA A6000 (Ampere).
# Include larger BLOCK_K and wider BLOCK_N to exploit Tensor Cores and high memory bandwidth
# for the very wide first/last layers and many 1024x1024 hidden layers.
# Conservative, explicit block-size choices to avoid long autotune compile times and resource overuse.
# These constants are passed as constexpr launch arguments at host-call time.
# Reduced DEFAULT_BLOCK_N (256) avoids shared-memory OOM on Ampere; increase DEFAULT_BLOCK_K
# to 64 to improve Tensor-Core utilization (multiple of 8). Keep DEFAULT_BLOCK_M conservative.
DEFAULT_BLOCK_M = 128
DEFAULT_BLOCK_N = 256  # reduced from 1024 to avoid shared-memory OOM on Ampere
DEFAULT_BLOCK_K = 64

@triton.jit
def _matmul_bias_relu_fp16_inputs_kernel(
    A_ptr,         # pointer to A (M, K) row-major, float16
    B_ptr,         # pointer to B (K, N) row-major, float16 (weight.t().half().contiguous())
    bias_ptr,      # pointer to bias (N,) float32
    C_ptr,         # pointer to output C (M, N) row-major, float16
    M, N, K,       # dimensions
    stride_am, stride_ak,    # strides for A (row-major)
    stride_bk, stride_bn,    # strides for B (row-major)
    stride_cm, stride_cn,    # strides for C (row-major)
    apply_relu,    # int flag: 1 to apply ReLU, 0 to skip
    BLOCK_M: tl.constexpr,   # block sizes are constexpr for autotuning
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Triton kernel optimized for Ampere:
      - A and B are float16 device buffers.
      - Uses float32 accumulator while performing tl.dot on float16 tiles.
      - Adds float32 bias and optionally applies ReLU in float32.
      - Writes float16 output.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute row and column ranges for this program
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # Initialize accumulator (float32)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    while k < K:
        offs_k = k + rk  # shape (BLOCK_K,)

        # A tile addresses: shape (BLOCK_M, BLOCK_K)
        A_tile_ptrs = A_ptr + (rm[:, None] * stride_am) + (offs_k[None, :] * stride_ak)
        mask_a = (rm[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(A_tile_ptrs, mask=mask_a, other=0.0)  # float16 expected in memory

        # B tile addresses: shape (BLOCK_K, BLOCK_N)
        B_tile_ptrs = B_ptr + (offs_k[:, None] * stride_bk) + (rn[None, :] * stride_bn)
        mask_b = (offs_k[:, None] < K) & (rn[None, :] < N)
        b = tl.load(B_tile_ptrs, mask=mask_b, other=0.0)  # float16 expected in memory

        # Multiply-accumulate (float16 inputs, accumulate into float32)
        acc += tl.dot(a, b)

        k += BLOCK_K

    # Add bias
    bias_ptrs = bias_ptr + rn
    bias_mask = rn < N
    bias_vec = tl.load(bias_ptrs, mask=bias_mask, other=0.0)
    acc = acc + bias_vec[None, :]

    # Optional ReLU in float32 accumulator
    if apply_relu != 0:
        acc = tl.where(acc > 0.0, acc, 0.0)

    # Cast to float16 and store
    out_half = acc.to(tl.float16)
    C_ptrs = C_ptr + (rm[:, None] * stride_cm) + (rn[None, :] * stride_cn)
    mask_c = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(C_ptrs, out_half, mask=mask_c)


def triton_linear_bias_relu_fp16_inputs(A_half: torch.Tensor, B_t_half: torch.Tensor, bias: torch.Tensor, apply_relu: bool):
    """
    Host wrapper for the Triton kernel with shared-memory safety clamping.

    Changes vs. previous:
      - Conservative initial heuristics for BLOCK_M, BLOCK_N, BLOCK_K tuned for A6000.
      - Compute the float32 accumulator footprint (BLOCK_M * BLOCK_N * 4 bytes) and clamp
        BLOCK_N first, then BLOCK_M, until the accumulator fits within device shared memory
        minus a safety margin. This avoids OutOfResources on Ampere GPUs.
      - Ensure BLOCK_K is a multiple of 8 for Tensor-Core friendliness.
      - Pass bias directly (no extra per-call contiguous copy); ModelNew keeps bias buffers up-to-date.
    """
    assert A_half.is_cuda and B_t_half.is_cuda and bias.is_cuda, "All tensors must be on CUDA."
    assert A_half.dtype == torch.half and B_t_half.dtype == torch.half and bias.dtype == torch.float32

    M, K = A_half.shape
    N = B_t_half.shape[1]

    # Allocate output in float16
    C = torch.empty((M, N), device=A_half.device, dtype=torch.half)

    # Strides (row-major)
    stride_am = A_half.stride(0)
    stride_ak = A_half.stride(1)
    stride_bk = B_t_half.stride(0)
    stride_bn = B_t_half.stride(1)
    stride_cm = C.stride(0)
    stride_cn = C.stride(1)

    # Conservative default choices (prefer smaller BLOCK_N for very wide N).
    BLOCK_K = DEFAULT_BLOCK_K if 'DEFAULT_BLOCK_K' in globals() else 64

    # Prefer reducing BLOCK_N for very wide outputs.
    if N >= 4096:
        BLOCK_N = 256
    elif N >= 1024:
        BLOCK_N = 256
    else:
        BLOCK_N = 128

    # Conservative BLOCK_M choices; will be clamped further if shared-memory would overflow.
    if M >= 256:
        BLOCK_M = 256
    elif M >= 128:
        BLOCK_M = 128
    else:
        BLOCK_M = 64

    # Shared memory safety check (A6000 approx shared memory limit).
    # Use a small safety margin to avoid hitting hardware limit.
    SHARED_LIMIT = 101376  # bytes (approx A6000)
    SAFETY_MARGIN = 2048   # bytes
    max_shared = SHARED_LIMIT - SAFETY_MARGIN

    # Compute accumulator size in bytes (float32 accumulator)
    acc_bytes = BLOCK_M * BLOCK_N * 4

    # Reduce BLOCK_N first (recommended by reviser guidance), then BLOCK_M, until we fit.
    # Prefer to keep BLOCK_M larger than BLOCK_N where possible, so reduce N first.
    while acc_bytes > max_shared and BLOCK_N > 64:
        BLOCK_N = BLOCK_N // 2
        acc_bytes = BLOCK_M * BLOCK_N * 4

    while acc_bytes > max_shared and BLOCK_M > 64:
        BLOCK_M = BLOCK_M // 2
        acc_bytes = BLOCK_M * BLOCK_N * 4

    # Final fallback to minimal safe tile if still too large.
    if acc_bytes > max_shared:
        BLOCK_M = 64
        BLOCK_N = 64
        acc_bytes = BLOCK_M * BLOCK_N * 4

    # Ensure BLOCK_K is a multiple of 8 for Tensor-Core friendliness.
    if BLOCK_K % 8 != 0:
        BLOCK_K = ((BLOCK_K + 7) // 8) * 8

    grid = ((M + BLOCK_M - 1) // BLOCK_M, (N + BLOCK_N - 1) // BLOCK_N)

    # Launch kernel using the clamped block sizes. Pass bias directly to avoid extra copies.
    _matmul_bias_relu_fp16_inputs_kernel[grid](
        A_half, B_t_half, bias, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        1 if apply_relu else 0,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return C

# Pre-compile / warm up the Triton kernel once if CUDA is available to avoid first-call compilation overhead.
# Use a deliberately small, safe kernel configuration for warmup so we do not compile an OOM configuration.
try:
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # Small warmup shapes that exercise the kernel path without much work.
        warm_M, warm_K, warm_N = 8, 8, 8
        A_w = torch.empty((warm_M, warm_K), device=device, dtype=torch.half)
        B_w = torch.empty((warm_K, warm_N), device=device, dtype=torch.half)
        bias_w = torch.empty((warm_N,), device=device, dtype=torch.float32)
        C_w = torch.empty((warm_M, warm_N), device=device, dtype=torch.half)

        # Strides for the warmup call
        stride_am = A_w.stride(0)
        stride_ak = A_w.stride(1)
        stride_bk = B_w.stride(0)
        stride_bn = B_w.stride(1)
        stride_cm = C_w.stride(0)
        stride_cn = C_w.stride(1)

        # Call the Triton kernel directly with very small constexpr block sizes to force
        # a safe compilation path and avoid compiling aggressive/heavy configs during import.
        _matmul_bias_relu_fp16_inputs_kernel[(1, 1)](
            A_w, B_w, bias_w, C_w,
            warm_M, warm_N, warm_K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            0,
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32
        )
        # Ensure compilation finishes.
        torch.cuda.synchronize()
except Exception:
    # If warmup fails (e.g., no CUDA at import or unexpected error), ignore and let runtime compile on first forward.
    pass

class ModelNew(nn.Module):
    """
    Optimized Model using Triton GEMM kernel:
      - Precomputes and registers transposed half-precision weight buffers for each Linear layer,
        and float32 bias buffers, to avoid repeated transposes and dtype conversions.
      - Converts activations to fp16 once at forward entry and keeps fp16 across layers.
      - Uses a high-throughput Triton kernel that performs tiled fp16 matmuls with fp32 accumulation,
        adds bias and optional ReLU, and writes fp16 activations for the next layer.
      - Converts final activation back to float32 to match original Model's output dtype.
    """
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        super(ModelNew, self).__init__()
        self.linears = nn.ModuleList()
        current_input_size = input_size

        # Build linear layers and register precomputed buffers (weight_t_half, bias_fp32)
        for i, hidden_size in enumerate(hidden_layer_sizes):
            lin = nn.Linear(current_input_size, hidden_size)
            self.linears.append(lin)
            # Register transposed half-weight buffer and bias buffer
            self.register_buffer(f'weight_t_half_{i}', lin.weight.t().contiguous().half())
            if lin.bias is not None:
                self.register_buffer(f'bias_fp32_{i}', lin.bias.contiguous().to(torch.float32))
            else:
                self.register_buffer(f'bias_fp32_{i}', torch.zeros((hidden_size,), dtype=torch.float32))
            current_input_size = hidden_size

        # final layer
        final = nn.Linear(current_input_size, output_size)
        self.linears.append(final)
        idx = len(hidden_layer_sizes)
        self.register_buffer(f'weight_t_half_{idx}', final.weight.t().contiguous().half())
        if final.bias is not None:
            self.register_buffer(f'bias_fp32_{idx}', final.bias.contiguous().to(torch.float32))
        else:
            self.register_buffer(f'bias_fp32_{idx}', torch.zeros((output_size,), dtype=torch.float32))

    def refresh_buffers(self):
        """
        Recompute the registered transposed-half-weight and bias buffers from current parameters.
        Call this if parameters have changed and you want the buffers to match the parameters.
        """
        for idx, linear in enumerate(self.linears):
            desired_w = linear.weight.t().contiguous().half()
            setattr(self, f'weight_t_half_{idx}', desired_w)
            if linear.bias is not None:
                desired_b = linear.bias.contiguous().to(torch.float32)
            else:
                desired_b = torch.zeros((linear.out_features,), dtype=torch.float32, device=linear.weight.device)
            setattr(self, f'bias_fp32_{idx}', desired_b.to(linear.weight.device))

    def _ensure_buffer_device_and_shape(self, idx):
        """
        Ensure registered buffers for layer idx are on correct device and have correct shapes.
        If not, refresh them from the corresponding nn.Linear parameter.
        """
        linear = self.linears[idx]
        buf_w_name = f'weight_t_half_{idx}'
        buf_b_name = f'bias_fp32_{idx}'

        buf_w = getattr(self, buf_w_name)
        desired_w = linear.weight.t().contiguous().half()
        if buf_w.shape != desired_w.shape or buf_w.device != desired_w.device:
            setattr(self, buf_w_name, desired_w)

        if linear.bias is not None:
            desired_b = linear.bias.contiguous().to(torch.float32)
        else:
            desired_b = torch.zeros((linear.out_features,), dtype=torch.float32, device=linear.weight.device)
        buf_b = getattr(self, buf_b_name)
        if buf_b.shape != desired_b.shape or buf_b.device != desired_b.device:
            setattr(self, buf_b_name, desired_b.to(linear.weight.device))

    def forward(self, x):
        """
        Forward pass:
          - Accepts x (batch_size, input_size) on CUDA. Converts to fp16 once and runs all layers
            using the Triton GEMM kernel.
          - Final output is converted back to float32.
        """
        if not x.is_cuda:
            raise RuntimeError("ModelNew.forward expects CUDA tensors (x must be on CUDA).")

        # Convert input to fp16 once (keep activations fp16 across layers)
        if x.dtype == torch.half:
            out = x.contiguous()
        else:
            out = x.contiguous().half()

        # Iterate layers and use Triton kernel for fused matmul + bias + optional ReLU
        # Collect registered buffers once to avoid repeated getattr/device checks in the loop.
        # Trust that buffers are maintained by init/refresh_buffers when parameters change.
        num_layers = len(self.linears)
        weights = [getattr(self, f'weight_t_half_{i}') for i in range(num_layers)]
        biases = [getattr(self, f'bias_fp32_{i}') for i in range(num_layers)]

        # Move activation to the weight device once (if needed). This avoids per-layer moves.
        weight_device = weights[0].device
        if out.device != weight_device:
            out = out.to(weight_device)

        # Move biases once if any are on a different device (cheap compared to doing this every layer).
        for i, b in enumerate(biases):
            if b.device != weight_device:
                biases[i] = b.to(weight_device)
                setattr(self, f'bias_fp32_{i}', biases[i])

        # Main loop uses the local lists and avoids per-iteration buffer/stride/device checks.
        for idx in range(num_layers):
            is_last = (idx == num_layers - 1)
            weight_t_half = weights[idx]
            bias_fp32 = biases[idx]

            # Call Triton helper: inputs are fp16 activation and fp16 transposed weight, bias is fp32
            out = triton_linear_bias_relu_fp16_inputs(out, weight_t_half, bias_fp32, apply_relu=not is_last)

        # Convert final fp16 activation back to float32 to match original interface
        return out.to(torch.float32)