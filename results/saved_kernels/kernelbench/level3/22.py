import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Cache folded (fp16 weight, fp32 bias) per (weight_ptr, bn_module) to avoid repeated folding.
# Keyed by (int(weight.data_ptr()), id(bn_module), device) so we don't attach attributes to Tensors.
_folded_cache = {}
# Cache output buffers and converted activation buffers per (device, M, N, dtype) to reduce repeated allocations
_out_cache = {}

# Fused GEMM kernel that can add a per-column bias and apply activation (ReLU / ReLU6)
# Autotune different tile sizes / warps for best performance on the target GPU.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=16),
        triton.Config({'BLOCK_M': 512, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=16),
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def _matmul_bias_act_kernel(
    A_ptr,  # pointer to A (row-major) shape (M, K) -> fp16 on host
    B_ptr,  # pointer to B (row-major) shape (K, N) -> fp16 on host
    C_ptr,  # pointer to C (row-major) shape (M, N) -> fp32
    bias_ptr,  # pointer to bias (length N) if APPLY_BIAS=1 (fp32) (ignored otherwise)
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    APPLY_BIAS: tl.constexpr,
    ACT_TYPE: tl.constexpr,  # 0 = none, 1 = ReLU, 2 = ReLU6
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # (BLOCK_M,)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # (BLOCK_N,)

    # Accumulator in fp32 for numeric stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Work in K-tiles; for each tile load the fp16 tiles and use them directly with tl.dot so tensor cores can be used.
    # Use a larger inner sub-block (SUB_K) aligned to tensor-core-friendly sizes.
    SUB_K = 16  # tensor-core-friendly chunk; ensure BLOCK_K in autotune configs is a multiple of SUB_K

    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)

        # Pointers to A and B tiles
        a_ptrs = A_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
        b_ptrs = B_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

        mask_a = (rm[:, None] < M) & (rk[None, :] < K)
        mask_b = (rk[:, None] < K) & (rn[None, :] < N)

        # Load tiles as fp16 (host-side stored); keep them fp16 so tl.dot can leverage Tensor Cores.
        a_h = tl.load(a_ptrs, mask=mask_a, other=0.0)  # (BLOCK_M, BLOCK_K), fp16
        b_h = tl.load(b_ptrs, mask=mask_b, other=0.0)  # (BLOCK_K, BLOCK_N), fp16

        # Vectorized block multiply over SUB_K chunks to reduce Python-level loop overhead and enable efficient matmuls.
        for kk in range(0, BLOCK_K, SUB_K):
            a_sub = a_h[:, kk:kk + SUB_K]        # (BLOCK_M, SUB_K), fp16
            b_sub = b_h[kk:kk + SUB_K, :]        # (SUB_K, BLOCK_N), fp16
            # Use fp16 inputs to tl.dot; acc is fp32 so accumulation remains in fp32 (enables mixed-precision Tensor Cores).
            acc += tl.dot(a_sub, b_sub)

    # Compute output pointer block (C is fp32)
    c_ptrs = C_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask_c = (rm[:, None] < M) & (rn[None, :] < N)

    out = acc
    if APPLY_BIAS:
        # load bias for columns rn (bias is fp32)
        mask_rn = rn < N
        bvec = tl.load(bias_ptr + rn, mask=mask_rn, other=0.0)  # (BLOCK_N,)
        out = out + bvec[None, :]

    # Activation (perform in fp32)
    if ACT_TYPE == 1:
        out = tl.maximum(out, 0.0)
    elif ACT_TYPE == 2:
        out = tl.minimum(tl.maximum(out, 0.0), 6.0)

    # Store fp32 outputs
    tl.store(c_ptrs, out, mask=mask_c)


def _ensure_2d_contiguous_mat(A: torch.Tensor):
    # Ensure tensor is 2D contiguous and float32 on CUDA
    if not A.is_cuda:
        raise RuntimeError("Triton fused path requires CUDA tensor.")
    if A.dtype != torch.float32:
        A = A.to(torch.float32)
    return A.contiguous()


def triton_1x1_conv_folded(
    input_tensor: torch.Tensor,
    conv_weight: torch.Tensor,
    bn_module: nn.BatchNorm2d = None,
    act_type: int = 0,
):
    """
    Perform 1x1 convolution via GEMM with optional folded BatchNorm and fused activation.
    Use mixed precision: activations & weights in fp16 for bandwidth savings, accumulation in fp32.

    Improvements:
    - Cache folded weight (fp16) and bias (fp32) per conv weight + BN module + device to avoid repeated folding.
    - Choose a representative DEFAULT_BLOCK_* for grid sizing using a small heuristic so autotuner configs
      respond better to large/small matrices.
    """
    if input_tensor.dim() != 4:
        raise ValueError("input_tensor must be 4D")
    B, C_in, H, W = input_tensor.shape

    if conv_weight.dim() == 4:
        C_out = conv_weight.shape[0]
        weight_flat = conv_weight.view(C_out, C_in).contiguous()
    elif conv_weight.dim() == 2:
        C_out, C_in_w = conv_weight.shape
        assert C_in_w == C_in, "Weight and input channel mismatch"
        weight_flat = conv_weight.contiguous()
    else:
        raise ValueError("conv_weight must be 2D or 4D tensor")

    # Prepare matrices for GEMM: A (M, K), Bmat (K, N)
    M = B * H * W
    K = C_in
    N = C_out

    # A: (M, K) as fp32 -> we'll cast to fp16 for the GEMM path
    A = input_tensor.permute(0, 2, 3, 1).contiguous().view(M, K)
    A = _ensure_2d_contiguous_mat(A)

    device = A.device

    # If BN folding possible (bn_module provided and in eval), fold weights & bias once and cache the result.
    if (bn_module is not None) and (not bn_module.training):
        # Use a cache key that encodes weight pointer, bn module id, and device to avoid repeated folding.
        key = (int(conv_weight.data_ptr()), id(bn_module), str(device))
        cached = _folded_cache.get(key, None)
        if cached is not None:
            Bmat = cached['Bmat']
            bias_tensor = cached['bias']
            APPLY_BIAS = 1
        else:
            dtype = A.dtype  # typically float32
            gamma = bn_module.weight.contiguous().to(device=device, dtype=dtype)
            beta = bn_module.bias.contiguous().to(device=device, dtype=dtype)
            running_mean = bn_module.running_mean.contiguous().to(device=device, dtype=dtype)
            running_var = bn_module.running_var.contiguous().to(device=device, dtype=dtype)
            eps = float(bn_module.eps)

            invstd = 1.0 / torch.sqrt(running_var + eps)  # (C_out,)
            scale = (gamma * invstd).view(C_out, 1)       # (C_out,1)
            weight_fold = (weight_flat.to(device=device, dtype=dtype) * scale)  # (C_out, C_in)
            bias_fold = beta - (gamma * invstd * running_mean)  # (C_out,)

            # Cast folded weight to fp16 for the GEMM path, keep bias in fp32
            Bmat = weight_fold.t().contiguous().half()  # (K, N) in fp16
            bias_tensor = bias_fold.contiguous().to(device=device, dtype=torch.float32)
            APPLY_BIAS = 1

            # Cache for reuse (store tensors on the correct device/dtype)
            _folded_cache[key] = {'Bmat': Bmat, 'bias': bias_tensor}
    else:
        # No folding: use weights as fp16 for GEMM, bias is dummy fp32
        Bmat = weight_flat.t().contiguous().half()  # (K, N) in fp16
        bias_tensor = torch.empty((1,), device=device, dtype=torch.float32)  # dummy
        APPLY_BIAS = 0

    # Convert A to fp16 to reduce bandwidth; kernel will cast to fp32 internally
    A_h = A.half().contiguous()  # (M, K), fp16

    # Output matrix in fp32 (accumulation + store in fp32 for correctness)
    # Try to reuse a cached output buffer to avoid repeated large allocations for repeated inference shapes.
    cache_key = (str(device), int(M), int(N), 'fp32')
    C = _out_cache.get(cache_key, None)
    if C is None or C.device != device or C.shape[0] != M or C.shape[1] != N:
        C = torch.empty((M, N), device=device, dtype=torch.float32)
        _out_cache[cache_key] = C

    # Strides (in elements) must match the dtypes we pass to the kernel
    stride_am = A_h.stride(0)
    stride_ak = A_h.stride(1)
    stride_bk = Bmat.stride(0)
    stride_bn = Bmat.stride(1)
    stride_cm = C.stride(0)
    stride_cn = C.stride(1)

    # Heuristic selection of representative block sizes to compute grid so autotuner picks fitting configurations.
    DEFAULT_BLOCK_M = 128
    DEFAULT_BLOCK_N = 128
    if M > 8 * 1024:
        DEFAULT_BLOCK_M = 256
    if M > 32 * 1024:
        DEFAULT_BLOCK_M = 512
    if N > 4 * 1024:
        DEFAULT_BLOCK_N = 256
    if N > 8 * 1024:
        DEFAULT_BLOCK_N = 512

    grid = (
        (M + DEFAULT_BLOCK_M - 1) // DEFAULT_BLOCK_M,
        (N + DEFAULT_BLOCK_N - 1) // DEFAULT_BLOCK_N,
    )

    # Launch autotuned kernel: A_h and Bmat are fp16, C and bias are fp32
    _matmul_bias_act_kernel[grid](
        A_h, Bmat, C, bias_tensor,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        APPLY_BIAS, act_type
    )

    # Return without forcing a contiguous copy to avoid the expensive permute+copy.
    out = C.view(B, H, W, N).permute(0, 3, 1, 2)
    return out


class MBConvNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super(MBConvNew, self).__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio
        self.expand_ratio = expand_ratio

        if expand_ratio != 1:
            self.expand_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
            self.expand_bn = nn.BatchNorm2d(hidden_dim)
            self.expand_relu = nn.ReLU6(inplace=True)
        else:
            self.expand_conv = None
            self.expand_bn = None
            self.expand_relu = None

        self.depthwise_conv = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride,
            padding=(kernel_size - 1) // 2, groups=hidden_dim, bias=False
        )
        self.depthwise_bn = nn.BatchNorm2d(hidden_dim)
        self.depthwise_relu = nn.ReLU6(inplace=True)

        self.project_conv = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x

        # Expand (1x1 conv) - try Triton folded BN path when on CUDA and BN is in eval mode
        if self.expand_conv is not None:
            if x.is_cuda and (not self.expand_bn.training):
                # Use triton path with BN folded and ReLU6 activation
                x = triton_1x1_conv_folded(x, self.expand_conv.weight, self.expand_bn, act_type=2)
            else:
                x = self.expand_conv(x)
                x = self.expand_bn(x)
                x = self.expand_relu(x)

        # Depthwise conv - use PyTorch (grouped conv is efficient on CUDA)
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x)
        x = self.depthwise_relu(x)

        # Project (1x1 conv) - fold BN, no activation after project (original MBConv)
        if x.is_cuda and (not self.project_bn.training):
            x = triton_1x1_conv_folded(x, self.project_conv.weight, self.project_bn, act_type=0)
        else:
            x = self.project_conv(x)
            x = self.project_bn(x)

        if self.use_residual:
            x = x + identity

        return x


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        EfficientNetB0-like architecture optimized with Triton-packed 1x1 convs
        where BatchNorm is folded into weights in eval mode and fused with activation.
        """
        super(ModelNew, self).__init__()

        # Initial convolutional layer (keep as PyTorch 3x3)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # MBConv blocks
        self.blocks = nn.Sequential(
            MBConvNew(32, 16, kernel_size=3, stride=1, expand_ratio=1),
            MBConvNew(16, 24, kernel_size=3, stride=2, expand_ratio=6),
            MBConvNew(24, 24, kernel_size=3, stride=1, expand_ratio=6),
            MBConvNew(24, 40, kernel_size=5, stride=2, expand_ratio=6),
            MBConvNew(40, 40, kernel_size=5, stride=1, expand_ratio=6),
            MBConvNew(40, 80, kernel_size=3, stride=2, expand_ratio=6),
            MBConvNew(80, 80, kernel_size=3, stride=1, expand_ratio=6),
            MBConvNew(80, 112, kernel_size=5, stride=1, expand_ratio=6),
            MBConvNew(112, 112, kernel_size=5, stride=1, expand_ratio=6),
            MBConvNew(112, 192, kernel_size=5, stride=2, expand_ratio=6),
            MBConvNew(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            MBConvNew(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            MBConvNew(192, 320, kernel_size=3, stride=1, expand_ratio=6),
        )

        # Final 1x1 conv + BN
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)

        # Fully connected layer
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        # Initial conv + BN + ReLU
        x = F.relu(self.bn1(self.conv1(x)))

        # MBConv blocks
        x = self.blocks(x)

        # Final 1x1 conv + BN + ReLU: use folded Triton path when possible
        if x.is_cuda and (not self.bn2.training):
            # act_type=1 -> ReLU
            x = triton_1x1_conv_folded(x, self.conv2.weight, self.bn2, act_type=1)
        else:
            x = F.relu(self.bn2(self.conv2(x)))

        # Global pooling + FC
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x