import torch
import torch.nn as nn
import torch.nn.functional as F

# Triton imports for the reduction kernel
import triton
import triton.language as tl

# Autotune configs tuned for NVIDIA A6000 (Ampere) with emphasis on wide channel tiles
# and spatial tiles matching typical final feature-map sizes (e.g., S=64 for 8x8).
AUTOTUNE_REDUCE = [
    triton.Config({"BLOCK_C": 512, "BLOCK_S": 64},  num_warps=16, num_stages=4),
    triton.Config({"BLOCK_C": 512, "BLOCK_S": 128}, num_warps=16, num_stages=4),
    triton.Config({"BLOCK_C": 256, "BLOCK_S": 64},  num_warps=8,  num_stages=3),
    triton.Config({"BLOCK_C": 256, "BLOCK_S": 128}, num_warps=8,  num_stages=3),
    triton.Config({"BLOCK_C": 128, "BLOCK_S": 256}, num_warps=8,  num_stages=3),
]


@triton.autotune(configs=AUTOTUNE_REDUCE, key=['B', 'C', 'S'])
@triton.jit
def _spatial_mean_kernel(
    x_ptr,       # pointer to input x flattened as (B, C, S)
    out_ptr,     # pointer to output means (B, C)
    B, C, S,     # integer args
    BLOCK_C: tl.constexpr, BLOCK_S: tl.constexpr,
):
    """
    Triton kernel to compute per-(B, C) spatial mean:
      out[b, c] = (1/S) * sum_s x[b, c, s]
    Streams across S in tiles (BLOCK_S) and tiles channels by BLOCK_C.
    Input: x flattened as (B, C, S), contiguous.
    Output: out flattened as (B, C), contiguous.
    """
    b = tl.program_id(0)
    c_block = tl.program_id(1)

    c_start = c_block * BLOCK_C
    offs_c = c_start + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C  # shape (BLOCK_C,)

    # accumulator per channel
    sum_per_c = tl.zeros((BLOCK_C,), dtype=tl.float32)

    s_block_start = 0
    # iterate over spatial tiles
    while s_block_start < S:
        s_start = s_block_start
        offs_s = s_start + tl.arange(0, BLOCK_S)
        mask_s = offs_s < S  # (BLOCK_S,)

        # addresses: ((b * C + c_idx) * S) + s_idx
        # c_idx shape: (BLOCK_C,1), s_idx shape: (1, BLOCK_S)
        c_idx = offs_c[:, None]          # (BLOCK_C, 1)
        s_idx = offs_s[None, :]          # (1, BLOCK_S)
        base = (b * C + c_idx) * S       # (BLOCK_C, 1)
        addrs = base + s_idx             # (BLOCK_C, BLOCK_S)
        mask = mask_c[:, None] & mask_s[None, :]  # (BLOCK_C, BLOCK_S)

        vals = tl.load(x_ptr + addrs, mask=mask, other=0.0)  # (BLOCK_C, BLOCK_S)
        # sum across spatial axis
        sum_s = tl.sum(vals, 1)  # (BLOCK_C,)
        sum_per_c += sum_s

        s_block_start += BLOCK_S

    invS = 1.0 / S
    mean_per_c = sum_per_c * invS  # (BLOCK_C,)

    out_addrs = b * C + offs_c
    tl.store(out_ptr + out_addrs, mean_per_c, mask=mask_c)


def fused_global_avgpool_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    """
    Two-stage fused operation optimized for Ampere:
      1) Run a Triton reduction kernel to compute spatial means per (B, C).
      2) Use torch.matmul (cuBLAS) for (B, C) @ (C, K) -> (B, K) with TF32 enabled
         for faster fp32 matmuls on Ampere devices.
    weight is nn.Linear.weight convention (K, C).
    """
    assert x.dtype == torch.float32 and weight.dtype == torch.float32
    if bias is not None:
        assert bias.dtype == torch.float32

    B, C, H, W = x.shape
    S = H * W
    K = weight.shape[0]

    # Flatten x to (B, C, S) contiguous
    x_flat = x.contiguous().view(B, C, S)

    # Allocate buffer for per-(B, C) means
    means = torch.empty((B, C), device=x.device, dtype=x.dtype)

    # Launch Triton reduction kernel
    grid = lambda meta: (B, (C + meta["BLOCK_C"] - 1) // meta["BLOCK_C"])
    _spatial_mean_kernel[grid](x_flat, means, B, C, S)

    # Prepare weight in (C, K) contiguous layout on the same device (avoid transposing every forward)
    weight_t = weight.t().contiguous().to(x.device)

    # Temporarily enable TF32 matmul on Ampere to use Tensor Cores for float32 GEMM
    enable_tf32 = False
    prev_matmul = None
    prev_cudnn = None
    if x.is_cuda:
        prev_matmul = torch.backends.cuda.matmul.allow_tf32
        prev_cudnn = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        enable_tf32 = True

    # GEMM using cuBLAS (torch.matmul)
    out = torch.matmul(means, weight_t)  # (B, K)

    # Restore TF32 flags to previous values
    if enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul
        torch.backends.cudnn.allow_tf32 = prev_cudnn

    # Add bias if provided
    if bias is not None:
        out += bias.view(1, -1).to(out.device)

    return out


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        EfficientNetB1-like architecture optimized for inference on Ampere GPUs.

        Optimizations:
          - Final global average pooling (spatial mean) is implemented in Triton as a high-throughput reduction.
          - The per-(B,C) means are consumed by a cuBLAS-backed matmul (torch.matmul) with TF32 enabled
            temporarily to exploit Tensor Cores on Ampere, yielding significant speedups for the FC layer.
          - Transposed FC weights (C, K) are cached per-device to avoid repeated transposes.
          - Utilities to fold BatchNorm into preceding Conv2d layers are provided for inference-time speedups.
        """
        super(ModelNew, self).__init__()

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # MBConv blocks
        self.mbconv1 = self._make_mbconv_block(32, 16, 1, 1)
        self.mbconv2 = self._make_mbconv_block(16, 24, 2, 6)
        self.mbconv3 = self._make_mbconv_block(24, 40, 2, 6)
        self.mbconv4 = self._make_mbconv_block(40, 80, 2, 6)
        self.mbconv5 = self._make_mbconv_block(80, 112, 1, 6)
        self.mbconv6 = self._make_mbconv_block(112, 192, 2, 6)
        self.mbconv7 = self._make_mbconv_block(192, 320, 1, 6)

        # Final convolutional layer
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)

        # Fully connected layer
        self.fc = nn.Linear(1280, num_classes)

        # Cache for transposed weight per device and underlying storage pointer
        # key: (device_index, weight_data_ptr)
        self._weight_t_cache = {}

    def _make_mbconv_block(self, in_channels, out_channels, stride, expand_ratio):
        hidden_dim = round(in_channels * expand_ratio)
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def _fuse_conv_bn(self, conv: nn.Conv2d, bn: nn.BatchNorm2d):
        """
        Fold batchnorm 'bn' into convolution 'conv' in-place for inference.
        """
        if not isinstance(conv, nn.Conv2d) or not isinstance(bn, nn.BatchNorm2d):
            return

        conv_w = conv.weight.data
        device = conv_w.device
        dtype = conv_w.dtype

        # Ensure conv has a bias tensor to write into
        if conv.bias is not None:
            conv_b = conv.bias.data
        else:
            conv_b = torch.zeros(conv_w.size(0), device=device, dtype=dtype)
            conv.bias = nn.Parameter(conv_b.clone())

        gamma = bn.weight.data.to(device=device, dtype=dtype)
        beta = bn.bias.data.to(device=device, dtype=dtype)
        running_mean = bn.running_mean.to(device=device, dtype=dtype)
        running_var = bn.running_var.to(device=device, dtype=dtype)
        eps = bn.eps

        denom = torch.sqrt(running_var + eps)
        scale = (gamma / denom)

        scale_w = scale.reshape(-1, 1, 1, 1)
        conv.weight.data.mul_(scale_w)
        conv.bias = nn.Parameter(((conv_b - running_mean) * scale + beta).clone())

    def fold_all_batchnorms(self):
        """
        Public helper to fold BatchNorm layers into preceding Conv2d layers.
        Call this when switching the model to evaluation/inference mode to reduce runtime.
        """
        for module in self.modules():
            children = list(module.named_children())
            for idx, (name, child) in enumerate(children):
                if isinstance(child, nn.Conv2d) and idx + 1 < len(children):
                    next_name, next_child = children[idx + 1]
                    if isinstance(next_child, nn.BatchNorm2d):
                        conv_mod = getattr(module, name)
                        bn_mod = getattr(module, next_name)
                        self._fuse_conv_bn(conv_mod, bn_mod)
                        setattr(module, next_name, nn.Identity())

        # Also try folding top-level common pairs
        top_pairs = [("conv1", "bn1"), ("conv2", "bn2")]
        for conv_name, bn_name in top_pairs:
            if hasattr(self, conv_name) and hasattr(self, bn_name):
                conv_mod = getattr(self, conv_name)
                bn_mod = getattr(self, bn_name)
                if isinstance(conv_mod, nn.Conv2d) and isinstance(bn_mod, nn.BatchNorm2d):
                    self._fuse_conv_bn(conv_mod, bn_mod)
                    setattr(self, bn_name, nn.Identity())

    def _get_cached_weight_t(self, device: torch.device):
        """
        Return cached (C, K) transposed weight on the given device.
        Cache keyed by (device_index, data_ptr) to detect weight updates.
        """
        device_idx = device.index if device.type == 'cuda' else -1
        weight_ptr = self.fc.weight.data_ptr()
        key = (device_idx, weight_ptr)

        cached = self._weight_t_cache.get(key, None)
        if cached is not None:
            if cached.device != device:
                cached = cached.to(device)
                self._weight_t_cache[key] = cached
            return cached

        # Build and cache transposed weight (C, K)
        with torch.no_grad():
            w_t = self.fc.weight.detach().t().contiguous().to(device)
            self._weight_t_cache[key] = w_t
            return w_t

    def forward(self, x):
        """
        Forward pass:
          - Convolutions and BatchNorms are executed by PyTorch.
          - Final global average pooling + linear is computed via a Triton reduction kernel
            to compute per-(B,C) means, followed by a cuBLAS-backed matmul (torch.matmul)
            with TF32 enabled on Ampere devices for improved throughput.
        """
        x = F.relu(self.bn1(self.conv1(x)))

        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        x = self.mbconv4(x)
        x = self.mbconv5(x)
        x = self.mbconv6(x)
        x = self.mbconv7(x)

        x = F.relu(self.bn2(self.conv2(x)))

        if x.is_cuda:
            # Use Triton reduction + cuBLAS matmul for best throughput on CUDA
            weight_t = self._get_cached_weight_t(x.device)  # (C, K)
            out = fused_global_avgpool_linear(x, weight_t.t(), self.fc.bias)
        else:
            # CPU fallback (pure PyTorch) for correctness
            x_avg = F.adaptive_avg_pool2d(x, (1, 1))
            x_flat = torch.flatten(x_avg, 1)
            out = self.fc(x_flat)

        return out


# Recreate the simple helper functions from the original file to be compatible with testing harnesses
batch_size = 10
input_shape = (3, 240, 240)
num_classes = 1000

def get_inputs():
    return [torch.rand(batch_size, *input_shape)]

def get_init_inputs():
    return [num_classes]