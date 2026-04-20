import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Autotune configs targeted for NVIDIA A6000 (Ampere) shapes encountered in VGG classifier.
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 8,   "BLOCK_N": 256, "BLOCK_K": 64},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 16,  "BLOCK_N": 256, "BLOCK_K": 64},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 16,  "BLOCK_N": 512, "BLOCK_K": 64},  num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 32,  "BLOCK_N": 512, "BLOCK_K": 64},  num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 32,  "BLOCK_N": 1024, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 512,  "BLOCK_K": 32}, num_warps=8, num_stages=3),
]

@triton.autotune(
    configs=AUTOTUNE_CONFIGS,
    key=["M", "N", "K"],
)
@triton.jit
def _triton_gemm_kernel(
    A_ptr,           # pointer to A (M, K)
    B_ptr,           # pointer to B (K, N)  -> we pass weight.t() so shape K x N
    C_ptr,           # pointer to output C (M, N)
    bias_ptr,        # pointer to bias (N) or any placeholder
    M,
    N,
    K,
    stride_am,       # stride of A along rows (elements)
    stride_ak,       # stride of A along cols (elements)
    stride_bk,       # stride of B along rows (elements)
    stride_bn,       # stride of B along cols (elements)
    stride_cm,       # stride of C along rows (elements)
    stride_cn,       # stride of C along cols (elements)
    RELU: tl.constexpr,    # whether to apply ReLU after bias add
    HAS_BIAS: tl.constexpr,# whether bias_ptr points to valid bias values
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D program ids selecting a tile of the output
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # row and column offsets for this program
    row_off = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # masks for valid rows/cols
    rm = row_off < M
    cm = col_off < N

    # accumulator for the tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # iterate over K dimension in chunks of BLOCK_K
    num_k_blocks = (K + BLOCK_K - 1) // BLOCK_K
    for bk in range(0, num_k_blocks):
        k0 = bk * BLOCK_K
        k_off = k0 + tl.arange(0, BLOCK_K)
        km = k_off < K

        # pointers for A and B blocks
        a_ptrs = A_ptr + (row_off[:, None] * stride_am) + (k_off[None, :] * stride_ak)
        b_ptrs = B_ptr + (k_off[:, None] * stride_bk) + (col_off[None, :] * stride_bn)

        mask_a = (rm[:, None]) & (km[None, :])
        mask_b = (km[:, None]) & (cm[None, :])

        # load blocks; shapes: a_block (BLOCK_M, BLOCK_K), b_block (BLOCK_K, BLOCK_N)
        a_block = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b_block = tl.load(b_ptrs, mask=mask_b, other=0.0)

        # vectorized multiply-accumulate over K-block
        # Use broadcasting multiplication and sum over K axis
        prod = a_block[:, :, None] * b_block[None, :, :]
        acc += tl.sum(prod, axis=1)

    # add bias if present
    if HAS_BIAS:
        bias_vals = tl.load(bias_ptr + col_off, mask=cm, other=0.0)
    else:
        bias_vals = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc = acc + bias_vals[None, :]

    # optional fused ReLU
    if RELU:
        acc = tl.maximum(acc, 0.0)

    # store the output tile
    c_ptrs = C_ptr + (row_off[:, None] * stride_cm) + (col_off[None, :] * stride_cn)
    mask_store = (rm[:, None]) & (cm[None, :])
    tl.store(c_ptrs, acc, mask=mask_store)


def triton_linear_from_wt(input: torch.Tensor, Wt: torch.Tensor, bias: torch.Tensor = None, relu: bool = False):
    """
    Triton-accelerated linear where the weight is provided already transposed (Wt == weight.t()).
    This avoids per-forward transposition overhead by allowing the forward pass to reuse a cached Wt.
    Expects CUDA tensors for Triton execution; falls back to PyTorch on CPU.
    """
    # Fallback to PyTorch if not CUDA to preserve correctness on CPU
    if not (input.is_cuda and Wt.is_cuda):
        # If Wt was provided, reconstruct weight by transposing
        W = Wt.t().contiguous()
        out = nn.functional.linear(input, W, bias)
        if relu:
            out = torch.relu(out)
        return out

    # ensure contiguity
    x = input.contiguous()
    Wt_c = Wt.contiguous()

    M = x.shape[0]
    K = x.shape[1]
    N = Wt_c.shape[1]  # because Wt is (K, N)

    # Prepare bias placeholder if None
    if bias is None:
        bias_tensor = torch.empty(1, dtype=x.dtype, device=x.device)
        has_bias = False
    else:
        bias_tensor = bias.contiguous()
        has_bias = True

    out = torch.empty((M, N), dtype=x.dtype, device=x.device)

    # Strides in elements (not bytes)
    stride_am = x.stride(0)
    stride_ak = x.stride(1)
    stride_bk = Wt_c.stride(0)
    stride_bn = Wt_c.stride(1)
    stride_cm = out.stride(0)
    stride_cn = out.stride(1)

    # Grid for 2D tiling
    grid = lambda meta: (
        (M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
        (N + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],
    )

    # Launch Triton kernel
    _triton_gemm_kernel[grid](
        x,                    # A_ptr
        Wt_c,                 # B_ptr (K, N)
        out,                  # C_ptr
        bias_tensor,          # bias_ptr
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        bool(relu),           # RELU constexpr
        bool(has_bias),       # HAS_BIAS constexpr
    )
    return out


class ModelNew(nn.Module):
    """
    VGG19 optimized for A6000:
      - Convolutional feature extractor computed in mixed precision (fp16) using torch.cuda.amp
        to leverage Tensor Cores for convolutions.
      - Classifier: three Linear layers kept as nn.Linear modules for state_dict compatibility,
        but during forward we reuse cached transposed weight tensors (Wt = weight.t().contiguous())
        to avoid repeated transpositions and accelerate Triton GEMM launches.
      - Triton GEMM fuses bias-add and optional ReLU.
    """
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()

        # Feature extractor (same as original VGG19)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Keep classifier parameters so weights/biases are visible in state_dict.
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.0),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.0),
            nn.Linear(4096, num_classes)
        )

        # Buffers to cache transposed weights for Triton usage.
        # They will be created lazily on the first forward when model is on CUDA.
        self._cached_wt0 = None
        self._cached_wt1 = None
        self._cached_wt2 = None

    def _ensure_cached_wt(self, linear_module, cache_name):
        """
        Ensure that cache_name attribute holds a contiguous transposed weight tensor
        on the same device/dtype as linear_module.weight. Create or refresh when needed.
        """
        cached = getattr(self, cache_name)
        w = linear_module.weight
        # If no cached tensor or device/dtype/shape mismatch, (re)create it
        if cached is None or cached.device != w.device or cached.dtype != w.dtype or cached.shape != (w.shape[1], w.shape[0]):
            # store transposed contiguous weight for reuse
            wt = w.t().contiguous()
            setattr(self, cache_name, wt)

    def forward(self, x):
        # Mixed precision for convolutional feature extractor on CUDA to leverage Tensor Cores.
        if x.is_cuda:
            with torch.cuda.amp.autocast(enabled=True):
                x = self.features(x)
        else:
            x = self.features(x)

        # Flatten features to (batch, 512*7*7)
        x = torch.flatten(x, 1)

        # Ensure classifier inputs are float32 for Triton GEMM (kernels assume fp32 loads/compute)
        if x.is_cuda:
            x = x.to(torch.float32)

        # Layer 1: Linear(512*7*7 -> 4096) + ReLU
        l0 = self.classifier[0]
        if x.is_cuda and l0.weight.is_cuda:
            # Ensure cached transposed weight exists on correct device/dtype
            self._ensure_cached_wt(l0, "_cached_wt0")
            x = triton_linear_from_wt(x, self._cached_wt0, l0.bias, relu=True)
        else:
            x = l0(x)
            x = torch.relu(x)

        # Layer 2: Linear(4096 -> 4096) + ReLU
        l1 = self.classifier[3]
        if x.is_cuda and l1.weight.is_cuda:
            self._ensure_cached_wt(l1, "_cached_wt1")
            x = triton_linear_from_wt(x, self._cached_wt1, l1.bias, relu=True)
        else:
            x = l1(x)
            x = torch.relu(x)

        # Layer 3: Linear(4096 -> num_classes) (no ReLU)
        l2 = self.classifier[6]
        if x.is_cuda and l2.weight.is_cuda:
            self._ensure_cached_wt(l2, "_cached_wt2")
            x = triton_linear_from_wt(x, self._cached_wt2, l2.bias, relu=False)
        else:
            x = l2(x)

        return x