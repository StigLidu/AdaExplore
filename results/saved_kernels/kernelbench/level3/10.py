import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Prefer cuDNN runtime heuristics for convolution selection on Ampere GPUs
# (low-risk global hint that improves conv kernel choices in many cases).
torch.backends.cudnn.benchmark = True

# Autotune configurations chosen for Ampere-like GPUs (A6000)
# Prefer BLOCK sizes that are multiples of common vector widths (128/256/512)
# for better coalescing and vectorized memory operations.
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 512}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK": 256}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK": 128}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK": 64},  num_warps=4, num_stages=2),
]


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['N', 'C', 'HW'])
@triton.jit
def _bn_affine_relu_kernel(x_ptr, s_ptr, b_ptr, out_ptr, N, C, HW, BLOCK: tl.constexpr):
    """
    Per-(sample,channel) tiled BN affine + ReLU (inference path).
    Grid: (N, C, ceil(HW / BLOCK))
    Each program handles one (n, c) and a spatial tile of size BLOCK.
    """
    n = tl.program_id(0)
    c = tl.program_id(1)
    block_idx = tl.program_id(2)

    offs = block_idx * BLOCK + tl.arange(0, BLOCK)
    mask = offs < HW

    base = ((n * C + c) * HW)
    ptr = x_ptr + base + offs

    x = tl.load(ptr, mask=mask, other=0.0)

    # load per-channel scale and bias once
    s = tl.load(s_ptr + c)
    b = tl.load(b_ptr + c)

    y = x * s + b
    # ReLU
    y = tl.where(y > 0.0, y, 0.0)

    tl.store(out_ptr + base + offs, y, mask=mask)


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['N', 'C', 'HW'])
@triton.jit
def _bn_affine_add_relu_kernel(conv_ptr, id_ptr, s_ptr, b_ptr, out_ptr, N, C, HW, BLOCK: tl.constexpr):
    """
    Per-(sample,channel) tiled BN affine + add(identity) + ReLU:
    Grid: (N, C, ceil(HW / BLOCK))
    """
    n = tl.program_id(0)
    c = tl.program_id(1)
    block_idx = tl.program_id(2)

    offs = block_idx * BLOCK + tl.arange(0, BLOCK)
    mask = offs < HW

    base = ((n * C + c) * HW)
    conv = tl.load(conv_ptr + base + offs, mask=mask, other=0.0)
    ident = tl.load(id_ptr + base + offs, mask=mask, other=0.0)

    s = tl.load(s_ptr + c)
    b = tl.load(b_ptr + c)

    y = conv * s + b + ident
    # ReLU
    y = tl.where(y > 0.0, y, 0.0)

    tl.store(out_ptr + base + offs, y, mask=mask)


# Small autotune configs for channel-tiling kernels (each program handles multiple channels)
# Expanded to include larger channel blocks and vector-friendly BLOCK sizes to improve
# memory coalescing and L2 reuse on Ampere.
AUTOTUNE_CONFIGS_CH = [
    triton.Config({"BLOCK": 512, "CHANNEL_BLOCK": 16}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK": 256, "CHANNEL_BLOCK": 16}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK": 256, "CHANNEL_BLOCK": 32}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK": 128, "CHANNEL_BLOCK": 32}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK": 128, "CHANNEL_BLOCK": 16}, num_warps=8, num_stages=2),
]

@triton.autotune(configs=AUTOTUNE_CONFIGS_CH, key=['N', 'C', 'HW'])
@triton.jit
def _bn_affine_relu_kernel_ch(x_ptr, s_ptr, b_ptr, out_ptr, N, C, HW, BLOCK: tl.constexpr, CHANNEL_BLOCK: tl.constexpr):
    """
    Channel-tiling variant: each program handles a small block of channels and a spatial tile.
    Grid: (N, ceil(C/CHANNEL_BLOCK), ceil(HW/BLOCK))
    """
    n = tl.program_id(0)
    ch_block = tl.program_id(1)
    block_idx = tl.program_id(2)

    offs = block_idx * BLOCK + tl.arange(0, BLOCK)
    c_offs = ch_block * CHANNEL_BLOCK + tl.arange(0, CHANNEL_BLOCK)

    mask_hw = offs < HW
    mask_c = c_offs < C
    mask = mask_hw[None, :] & mask_c[:, None]

    base = (n * C) * HW + c_offs * HW  # shape [CHANNEL_BLOCK]
    ptr = x_ptr + base[:, None] + offs[None, :]
    x = tl.load(ptr, mask=mask, other=0.0)  # shape [CHANNEL_BLOCK, BLOCK]

    s = tl.load(s_ptr + c_offs, mask=mask_c, other=0.0)[:, None]
    b = tl.load(b_ptr + c_offs, mask=mask_c, other=0.0)[:, None]

    y = x * s + b
    # ReLU
    y = tl.where(y > 0.0, y, 0.0)

    tl.store(out_ptr + base[:, None] + offs[None, :], y, mask=mask)


@triton.autotune(configs=AUTOTUNE_CONFIGS_CH, key=['N', 'C', 'HW'])
@triton.jit
def _bn_affine_add_relu_kernel_ch(conv_ptr, id_ptr, s_ptr, b_ptr, out_ptr, N, C, HW, BLOCK: tl.constexpr, CHANNEL_BLOCK: tl.constexpr):
    """
    Channel-tiling variant of BN affine + add(identity) + ReLU.
    Grid: (N, ceil(C/CHANNEL_BLOCK), ceil(HW/BLOCK))
    """
    n = tl.program_id(0)
    ch_block = tl.program_id(1)
    block_idx = tl.program_id(2)

    offs = block_idx * BLOCK + tl.arange(0, BLOCK)
    c_offs = ch_block * CHANNEL_BLOCK + tl.arange(0, CHANNEL_BLOCK)

    mask_hw = offs < HW
    mask_c = c_offs < C
    mask = mask_hw[None, :] & mask_c[:, None]

    base = (n * C) * HW + c_offs * HW  # shape [CHANNEL_BLOCK]
    conv = tl.load(conv_ptr + base[:, None] + offs[None, :], mask=mask, other=0.0)
    ident = tl.load(id_ptr + base[:, None] + offs[None, :], mask=mask, other=0.0)

    s = tl.load(s_ptr + c_offs, mask=mask_c, other=0.0)[:, None]
    b = tl.load(b_ptr + c_offs, mask=mask_c, other=0.0)[:, None]

    y = conv * s + b + ident
    # ReLU
    y = tl.where(y > 0.0, y, 0.0)

    tl.store(out_ptr + base[:, None] + offs[None, :], y, mask=mask)


def _compute_bn_affine_params(bn: nn.BatchNorm2d, device, dtype):
    """
    Compute scale and bias for BN inference:
      s = gamma / sqrt(running_var + eps)
      b = beta - running_mean * s
    """
    gamma = bn.weight
    beta = bn.bias
    running_mean = bn.running_mean
    running_var = bn.running_var
    eps = bn.eps
    s = (gamma / torch.sqrt(running_var + eps)).to(device=device, dtype=dtype).contiguous()
    b = (beta - running_mean * s).to(device=device, dtype=dtype).contiguous()
    return s, b


def _fold_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d, device, dtype):
    """
    Fold BatchNorm affine into conv weights and bias for inference:
      W' = W * s.reshape(out_channels,1,1,1)
      b' = (conv.bias if present else 0) * s + b
    Returns (W_folded, b_folded) on device/dtype.

    To support a safe fp16 TensorCore path, compute the BN coefficients (s, b)
    in float32 for numerical stability, then cast the folded weights/bias to
    the requested dtype (e.g., float16) before returning.
    """
    # detach original params (do not modify originals)
    W = conv.weight.detach()
    conv_bias = conv.bias.detach() if conv.bias is not None else None

    gamma = bn.weight.detach()
    beta = bn.bias.detach()
    running_mean = bn.running_mean
    running_var = bn.running_var
    eps = bn.eps

    # compute per-channel scale and bias in FP32 to preserve accuracy
    s_fp32 = (gamma / torch.sqrt(running_var + eps)).to(device=device, dtype=torch.float32)
    b_fp32 = (beta - running_mean * s_fp32).to(device=device, dtype=torch.float32)

    # cast per requested dtype for folding (weights/bias)
    if dtype == torch.float16:
        s = s_fp32.to(device=device, dtype=torch.float16).contiguous()
        b = b_fp32.to(device=device, dtype=torch.float16).contiguous()
        W = W.to(device=device, dtype=torch.float16).contiguous()
    else:
        s = s_fp32.to(device=device, dtype=dtype).contiguous()
        b = b_fp32.to(device=device, dtype=dtype).contiguous()
        W = W.to(device=device, dtype=dtype).contiguous()

    W_fold = W * s.reshape(-1, 1, 1, 1)
    if conv_bias is not None:
        b_fold = conv_bias.to(device=device, dtype=dtype).contiguous() * s + b
    else:
        b_fold = b

    return W_fold, b_fold


def _get_or_make_folded(conv: nn.Conv2d, bn: nn.BatchNorm2d, device, dtype):
    """
    Get cached folded (W, b) for given conv+bn on (device, dtype) or compute and cache it.
    The cache is stored on the conv module as attribute _fold_cache (a dict keyed by (device, dtype)).
    """
    cache = getattr(conv, "_fold_cache", None)
    key = (device, dtype)
    if cache is None:
        cache = {}
        setattr(conv, "_fold_cache", cache)
    if key in cache:
        return cache[key]
    W_fold, b_fold = _fold_conv_bn(conv, bn, device, dtype)
    # keep a reference to the GPU tensor on the conv module via the cache dict
    cache[key] = (W_fold, b_fold)
    return W_fold, b_fold


def _bn_affine_relu_triton(x: torch.Tensor, bn: nn.BatchNorm2d):
    """
    Apply BatchNorm (inference) then ReLU using Triton.
    For inference & CUDA & no grad: operate in-place on x's buffer to minimize memory traffic.
    Falls back to PyTorch if training or requires_grad or not CUDA.
    """
    if x.requires_grad:
        return F.relu(bn(x))

    if (not x.is_cuda) or bn.training:
        return F.relu(bn(x))

    device = x.device
    dtype = x.dtype
    with torch.no_grad():
        s, b = _compute_bn_affine_params(bn, device, dtype)

        # avoid unnecessary device->device copies if already contiguous
        x_contig = x if x.is_contiguous() else x.contiguous()
        N, C, H, W = x_contig.shape
        HW = H * W

        # in-place (write back to same buffer)
        grid = lambda meta: (N, C, (HW + meta['BLOCK'] - 1) // meta['BLOCK'])
        _bn_affine_relu_kernel[grid](x_contig, s, b, x_contig, N, C, HW)
        return x_contig


def _bn_affine_add_relu_triton(conv_out: torch.Tensor, identity: torch.Tensor, bn: nn.BatchNorm2d):
    """
    Apply BN affine to conv_out (inference) + identity + ReLU via Triton in-place on conv_out buffer.
    Falls back to PyTorch for training/autograd/CPU cases.
    """
    if conv_out.requires_grad or identity.requires_grad:
        return F.relu(bn(conv_out) + identity)

    if (not conv_out.is_cuda) or (not identity.is_cuda) or bn.training:
        return F.relu(bn(conv_out) + identity)

    device = conv_out.device
    dtype = conv_out.dtype
    with torch.no_grad():
        s, b = _compute_bn_affine_params(bn, device, dtype)
        # avoid unnecessary device->device copies if already contiguous
        x_contig = conv_out if conv_out.is_contiguous() else conv_out.contiguous()
        id_contig = identity if identity.is_contiguous() else identity.contiguous()

        N, C, H, W = x_contig.shape
        HW = H * W

        grid = lambda meta: (N, C, (HW + meta['BLOCK'] - 1) // meta['BLOCK'])
        _bn_affine_add_relu_kernel[grid](x_contig, id_contig, s, b, x_contig, N, C, HW)
        return x_contig


class BottleneckNew(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleneckNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        # downsample is usually nn.Sequential(conv, bn)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        # Fast inference path: fold conv+bn into single convs and use cuDNN (F.conv2d) for convolutions.
        # This minimizes memory passes (no separate BN step) and leverages optimized conv kernels.
        if (not self.training) and (not x.requires_grad) and x.is_cuda:
            device = x.device
            dtype = x.dtype

            # Prefer channels-last activations so cuDNN can use fast NHWC/TensorCore paths on Ampere.
            # This is safe in the inference-only branch and avoids autograd concerns.
            x = x.contiguous(memory_format=torch.channels_last)
            identity = identity.contiguous(memory_format=torch.channels_last)

            # conv1 folded (cached)
            w1, b1 = _get_or_make_folded(self.conv1, self.bn1, device, dtype)
            out = F.conv2d(x, w1, bias=b1, stride=1, padding=0)
            out = F.relu(out)

            # conv2 folded (preserve stride and padding) (cached)
            w2, b2 = _get_or_make_folded(self.conv2, self.bn2, device, dtype)
            out = F.conv2d(out, w2, bias=b2, stride=self.stride, padding=1)
            out = F.relu(out)

            # conv3 folded (cached)
            w3, b3 = _get_or_make_folded(self.conv3, self.bn3, device, dtype)
            out = F.conv2d(out, w3, bias=b3, stride=1, padding=0)

            # fold downsample if present (cached)
            if self.downsample is not None:
                ds_conv = self.downsample[0]
                ds_bn = self.downsample[1]
                wds, bds = _get_or_make_folded(ds_conv, ds_bn, device, dtype)
                identity = F.conv2d(x, wds, bias=bds, stride=self.stride, padding=0)

            out = out + identity
            out = F.relu(out)
            return out

        # Training / autograd / CPU path: keep original semantics, but use Triton fusions where safe in inference-like cases.
        out = self.conv1(x)
        out = _bn_affine_relu_triton(out, self.bn1)

        out = self.conv2(out)
        out = _bn_affine_relu_triton(out, self.bn2)

        out = self.conv3(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out = _bn_affine_add_relu_triton(out, identity, self.bn3)
        return out


class ModelNew(nn.Module):
    def __init__(self, layers, num_classes=1000):
        super(ModelNew, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        block = BottleneckNew

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Fast inference path: fold conv1 + bn1 when possible to use a single conv
        if (not self.training) and (not x.requires_grad) and x.is_cuda:
            device = x.device
            dtype = x.dtype
            # Prefer channels-last activations to let cuDNN pick NHWC/TensorCore kernels on Ampere.
            x = x.contiguous(memory_format=torch.channels_last)
            w1, b1 = _get_or_make_folded(self.conv1, self.bn1, device, dtype)
            x = F.conv2d(x, w1, bias=b1, stride=2, padding=3)
            x = F.relu(x)
        else:
            x = self.conv1(x)
            x = _bn_affine_relu_triton(x, self.bn1)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# Utilities to match original module signature (input generation helpers)
batch_size = 10
height = 224
width = 224
layers = [3, 4, 23, 3]
num_classes = 1000

def get_inputs():
    return [torch.rand(batch_size, 3, height, width)]

def get_init_inputs():
    return [layers, num_classes]