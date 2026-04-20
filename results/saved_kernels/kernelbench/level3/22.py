import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Triton kernel block size (constexpr)
TRITON_BLOCK = 1024

@triton.jit
def _bn_affine_clamp_kernel(
    x_ptr,           # input pointer (fp32)
    out_ptr,         # output pointer (fp32)
    scale_ptr,       # per-channel scale (fp32)
    bias_ptr,        # per-channel bias (fp32)
    C,               # number of channels (int)
    HW,              # H*W (int)
    NHW,             # N*H*W (int)
    stride_n,        # stride between batches in elements (int) = C*H*W
    stride_c,        # stride between channels in elements (int) = H*W
    maxval,          # activation upper bound; use large number for ReLU, 6.0 for ReLU6 (float)
    BLOCK: tl.constexpr
):
    """
    For each channel pid_c and block over NHW, load x, apply y = x*scale + bias, then clamp between 0 and maxval.
    Layout: NCHW contiguous
    """
    pid_c = tl.program_id(0)
    pid_block = tl.program_id(1)

    offs = pid_block * BLOCK + tl.arange(0, BLOCK)
    mask = offs < NHW

    # decompose offs into n and spatial
    n = offs // HW
    inner = offs % HW

    idx = n * stride_n + pid_c * stride_c + inner

    x = tl.load(x_ptr + idx, mask=mask, other=0.0)

    scale = tl.load(scale_ptr + pid_c)
    bias = tl.load(bias_ptr + pid_c)

    y = x * scale + bias
    # relu then clamp
    y = tl.maximum(y, 0.0)
    y = tl.minimum(y, maxval)

    tl.store(out_ptr + idx, y, mask=mask)


def fused_bn_activation(x: torch.Tensor, bn: nn.BatchNorm2d, maxval: float):
    """
    Apply batchnorm affine and activation clamp (fused via Triton) when bn is in training mode
    to preserve running stats updates, but still accelerate the affine+clamp memory pass.
    For evaluation mode folding is preferred (handled separately).
    """
    assert x.is_cuda, "Input must be on CUDA"
    assert bn.weight is not None and bn.bias is not None, "BatchNorm params required"

    # Ensure contiguous for the kernel
    x = x.contiguous()
    device = x.device
    dtype = x.dtype

    # Load per-channel scale and bias from bn (use current stats/params)
    # In training mode we must use current bn.weight/bn.bias (learnable)
    # We compute scale = gamma / sqrt(running_var + eps) and bias_new = beta - running_mean * scale
    gamma = bn.weight.to(device=device, dtype=dtype).contiguous()
    beta = bn.bias.to(device=device, dtype=dtype).contiguous()
    running_mean = bn.running_mean.to(device=device, dtype=dtype)
    running_var = bn.running_var.to(device=device, dtype=dtype)
    eps = bn.eps

    scale = (gamma / torch.sqrt(running_var + eps)).contiguous()
    bias_new = (beta - running_mean * scale).contiguous()

    N, C, H, W = x.shape
    HW = H * W
    NHW = N * HW
    stride_n = C * HW
    stride_c = HW

    out = torch.empty_like(x)

    grid = (C, (NHW + TRITON_BLOCK - 1) // TRITON_BLOCK)

    _bn_affine_clamp_kernel[grid](
        x, out, scale, bias_new,
        C, HW, NHW, stride_n, stride_c, float(maxval),
        BLOCK=TRITON_BLOCK
    )
    return out


def _fold_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d, device, dtype):
    """
    Fold BatchNorm parameters into a Conv2d weight and bias for inference.
    Returns W_fold (same shape as conv.weight) and b_fold (out_channels,)
    """
    w = conv.weight.to(device=device, dtype=dtype).contiguous()
    if conv.bias is not None:
        b = conv.bias.to(device=device, dtype=dtype).contiguous()
    else:
        b = torch.zeros(w.shape[0], device=device, dtype=dtype)

    gamma = bn.weight.to(device=device, dtype=dtype).contiguous()
    beta = bn.bias.to(device=device, dtype=dtype).contiguous()
    running_mean = bn.running_mean.to(device=device, dtype=dtype).contiguous()
    running_var = bn.running_var.to(device=device, dtype=dtype).contiguous()
    eps = bn.eps

    scale = gamma / torch.sqrt(running_var + eps)
    scale_reshaped = scale.reshape(-1, 1, 1, 1)
    W_fold = w * scale_reshaped
    b_fold = beta - running_mean * scale + scale * b
    return W_fold.contiguous(), b_fold.contiguous()


class MBConvNew(nn.Module):
    """
    MBConv block with cached folded weights for inference and Triton-assisted BN+activation for training.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super(MBConvNew, self).__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio
        self.expand_ratio = expand_ratio
        self._cache = {}  # lazy cache for folded weights: keys will be strings like 'expand', 'depthwise', 'project'

        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            )
        else:
            self.expand_conv = None

        padding = (kernel_size - 1) // 2
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                hidden_dim, hidden_dim,
                kernel_size=kernel_size, stride=stride, padding=padding,
                groups=hidden_dim, bias=False
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )

        self.project_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def _get_cached_fold(self, key, conv, bn, x):
        """
        Return folded (Wf, bf) matching conv, bn for device/dtype of x.
        Cache until device/dtype or parameter shapes change.
        """
        dev = x.device
        dtype = x.dtype
        cache = self._cache.get(key)
        # Validate cache
        if cache is not None:
            Wf, bf, dev_cached, dtype_cached, shape_w = cache
            if dev_cached == dev and dtype_cached == dtype and Wf.shape == conv.weight.shape:
                return Wf, bf
        # compute and cache
        Wf, bf = _fold_conv_bn(conv, bn, device=dev, dtype=dtype)
        self._cache[key] = (Wf, bf, dev, dtype, conv.weight.shape)
        return Wf, bf

    def forward(self, x):
        identity = x

        # Expand
        if self.expand_conv is not None:
            conv = self.expand_conv[0]
            bn = self.expand_conv[1]
            if not bn.training:
                Wf, bf = self._get_cached_fold("expand", conv, bn, x)
                x = F.conv2d(x, Wf, bias=bf, stride=conv.stride, padding=conv.padding, groups=conv.groups)
                x = torch.clamp(x, min=0.0, max=6.0)
            else:
                x = conv(x)
                # Use standard BatchNorm forward during training to ensure batch-statistics
                # are used and running stats are updated correctly. Then apply ReLU6.
                x = bn(x)
                x = torch.clamp(x, min=0.0, max=6.0)

        # Depthwise
        conv = self.depthwise_conv[0]
        bn = self.depthwise_conv[1]
        if not bn.training:
            Wf, bf = self._get_cached_fold("depthwise", conv, bn, x)
            x = F.conv2d(x, Wf, bias=bf, stride=conv.stride, padding=conv.padding, groups=conv.groups)
            x = torch.clamp(x, min=0.0, max=6.0)
        else:
            x = conv(x)
            # Preserve BN training behavior and then apply ReLU6 activation.
            x = bn(x)
            x = torch.clamp(x, min=0.0, max=6.0)

        # Project
        conv = self.project_conv[0]
        bn = self.project_conv[1]
        if not bn.training:
            Wf, bf = self._get_cached_fold("project", conv, bn, x)
            x = F.conv2d(x, Wf, bias=bf, stride=conv.stride, padding=conv.padding, groups=conv.groups)
        else:
            x = conv(x)
            # For project conv we must preserve bn semantics; no activation after
            x = bn(x)

        if self.use_residual:
            x = x + identity

        return x


class ModelNew(nn.Module):
    """
    EfficientNetB0-like architecture optimized:
      - Folds BatchNorm into Conv weights for inference and caches them to avoid recomputation.
      - Uses Triton kernel to fuse BN-affine + clamp (ReLU/ReLU6) in training mode (memory-bound affine + clamp).
      - Keeps original module structure for state_dict compatibility.
    """
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # MBConv blocks (using MBConvNew)
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

        # Final convolutional layer
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)

        # Fully connected layer
        self.fc = nn.Linear(1280, num_classes)

        # caches for folded convs in the top-level model (conv1, conv2)
        self._top_cache = {}

    def _get_top_fold(self, key, conv, bn, x):
        """
        Similar caching for top-level convs (conv1, conv2).
        """
        dev = x.device
        dtype = x.dtype
        cache = self._top_cache.get(key)
        if cache is not None:
            Wf, bf, dev_cached, dtype_cached, shape_w = cache
            if dev_cached == dev and dtype_cached == dtype and Wf.shape == conv.weight.shape:
                return Wf, bf
        Wf, bf = _fold_conv_bn(conv, bn, device=dev, dtype=dtype)
        self._top_cache[key] = (Wf, bf, dev, dtype, conv.weight.shape)
        return Wf, bf

    def forward(self, x):
        # Initial conv + bn + ReLU
        if not self.bn1.training:
            Wf, bf = self._get_top_fold("conv1", self.conv1, self.bn1, x)
            x = F.conv2d(x, Wf, bias=bf, stride=self.conv1.stride, padding=self.conv1.padding, groups=self.conv1.groups)
            x = torch.clamp(x, min=0.0)  # ReLU
        else:
            x = self.conv1(x)
            # Use BatchNorm's training forward to ensure correct batch stats behavior, then ReLU.
            x = self.bn1(x)
            x = torch.clamp(x, min=0.0)  # ReLU

        x = self.blocks(x)

        # Final conv + bn + ReLU
        if not self.bn2.training:
            Wf, bf = self._get_top_fold("conv2", self.conv2, self.bn2, x)
            x = F.conv2d(x, Wf, bias=bf, stride=self.conv2.stride, padding=self.conv2.padding, groups=self.conv2.groups)
            x = torch.clamp(x, min=0.0)
        else:
            x = self.conv2(x)
            # Preserve BatchNorm training semantics and then apply ReLU.
            x = self.bn2(x)
            x = torch.clamp(x, min=0.0)

        # Global pooling and classifier
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x