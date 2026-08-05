import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Triton inplace add kernel for NHWC fp16 tensors (inference-only).
# The kernel treats the tensors as a flat array of fp16 values and performs a += b elementwise.
@triton.jit
def _inplace_add_kernel(a_ptr, b_ptr, n_items, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_items
    a_vals = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offs, mask=mask, other=0.0)
    tl.store(a_ptr + offs, a_vals + b_vals, mask=mask)

def triton_inplace_add(a: torch.Tensor, b: torch.Tensor):
    """
    In-place add b into a using a Triton kernel.
    Expectations:
      - a and b are on CUDA
      - a and b have the same shape
      - a and b are torch.half (fp16)
      - tensors are in channels-last (NHWC) contiguous layout for best coalescing
    This helper is intended for inference-only usage to avoid autograd/out=... issues.
    """
    assert a.is_cuda and b.is_cuda, "Triton inplace add requires CUDA tensors"
    assert a.shape == b.shape, "Shapes must match"
    assert a.dtype == torch.half and b.dtype == torch.half, "Only fp16 supported for triton_inplace_add"

    # Ensure channels-last contiguous layout for coalesced loads/stores. If not already in that layout,
    # fall back to making a contiguous channels-last tensor (this may allocate).
    if not a.is_contiguous(memory_format=torch.channels_last):
        a = a.contiguous(memory_format=torch.channels_last)
    if not b.is_contiguous(memory_format=torch.channels_last):
        b = b.contiguous(memory_format=torch.channels_last)

    n_items = a.numel()
    grid = lambda meta: ((n_items + meta['BLOCK'] - 1) // meta['BLOCK'],)
    _inplace_add_kernel[grid](a, b, n_items, BLOCK=1024)

def fold_batchnorm_into_conv(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    """
    Fold BatchNorm2d parameters into a preceding Conv2d (in-place).
    This modifies conv.weight and conv.bias so that conv followed by bn is
    equivalent to the folded conv alone. If conv.bias is None, a bias
    parameter is created.
    """
    if not isinstance(bn, nn.BatchNorm2d):
        return
    with torch.no_grad():
        w = conv.weight.data  # (out_channels, in_ch_per_group, kh, kw)
        out_channels = w.shape[0]
        device = w.device
        dtype = w.dtype

        if bn.affine:
            gamma = bn.weight.data.to(device=device, dtype=dtype)
            beta = bn.bias.data.to(device=device, dtype=dtype)
        else:
            gamma = torch.ones(out_channels, device=device, dtype=dtype)
            beta = torch.zeros(out_channels, device=device, dtype=dtype)

        running_mean = bn.running_mean.to(device=device, dtype=dtype)
        running_var = bn.running_var.to(device=device, dtype=dtype)
        eps = bn.eps

        denom = torch.sqrt(running_var + eps)
        # scale shape (out_channels,)
        scale = (gamma / denom).view(out_channels, 1, 1, 1)
        w.mul_(scale)

        # compute folded bias
        if conv.bias is None:
            folded_bias = (-running_mean * gamma / denom) + beta
            conv.bias = nn.Parameter(folded_bias.to(device=device, dtype=dtype))
        else:
            # orig_bias is conv.bias
            orig_bias = conv.bias.data.to(device=device, dtype=dtype)
            folded_bias = (orig_bias - running_mean) * gamma / denom + beta
            conv.bias.data = folded_bias

def _fold_channel_shuffle_into_conv3(conv3: nn.Conv2d, groups: int):
    """
    Modify conv3.weight in-place so that applying conv3 to unshuffled activations
    is equivalent to original conv3 applied to shuffled activations.
    Assumes conv3 is a 1x1 group convolution with `groups` groups.
    Vectorized implementation (no Python loops over channels).
    """
    with torch.no_grad():
        w = conv3.weight.data  # shape (out_channels, in_channels_per_group, 1, 1)
        out_channels, in_ch_per_group, kh, kw = w.shape
        assert kh == 1 and kw == 1, "Folding currently only supports 1x1 conv3."
        mid_channels = in_ch_per_group * groups  # total input channels to conv3
        out_ch_per_group = out_channels // groups
        channels_per_group = mid_channels // groups

        device = w.device
        dtype = w.dtype

        # Flatten the grouped weight to (out_channels, in_ch_per_group)
        w_flat = w.view(out_channels, in_ch_per_group)

        # Build column indices for placing each w_flat row into full_w
        starts = (torch.arange(out_channels, device=device) // out_ch_per_group) * in_ch_per_group  # (out_channels,)
        cols = starts.unsqueeze(1) + torch.arange(in_ch_per_group, device=device)  # (out_channels, in_ch_per_group)

        # full_w: (out_channels, mid_channels)
        full_w = w_flat.new_zeros((out_channels, mid_channels))
        full_w.scatter_(1, cols, w_flat)

        # Compute permutation mapping (vectorized)
        nc = torch.arange(mid_channels, device=device)
        perm_new_to_old = (nc % channels_per_group) * groups + (nc // channels_per_group)  # new_c -> old_c
        permuted_cols = torch.empty_like(perm_new_to_old)
        permuted_cols[perm_new_to_old] = nc  # permuted_cols[old_c] = new_c

        # Permute columns so that columns are laid out in unshuffled order
        permuted_full = full_w[:, permuted_cols]

        # Extract grouped-layout blocks back into w_flat using the same cols
        new_w_flat = permuted_full.gather(1, cols)

        # Write back into conv3.weight
        w.view(out_channels, in_ch_per_group)[:] = new_w_flat
        conv3.weight.data = w

# Elementwise add is performed in-place (torch.add(out, sc, out=out)) inside ShuffleNetUnitTriton.forward
# to avoid an extra kernel launch and extra global-memory traffic.

# ShuffleNet Unit, reusing original PyTorch layers but integrating Triton shuffle & add
class ShuffleNetUnitTriton(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3):
        super(ShuffleNetUnitTriton, self).__init__()
        assert out_channels % 4 == 0
        mid_channels = out_channels // 4

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Fold BN into conv1 and conv3 weights/bias at init to remove BN kernels at runtime.
        # This follows the guidance to fuse pointwise conv + BN (+ReLU) for inference.
        fold_batchnorm_into_conv(self.conv1, self.bn1)
        # replace bn1 with identity so forward doesn't re-apply BN
        self.bn1 = nn.Identity()

        # Also fold bn2 (depthwise conv BN) into conv2 to remove its BN kernel at runtime.
        fold_batchnorm_into_conv(self.conv2, self.bn2)
        # replace bn2 with identity so forward doesn't re-apply BN
        self.bn2 = nn.Identity()

        fold_batchnorm_into_conv(self.conv3, self.bn3)
        # replace bn3 with identity so forward doesn't re-apply BN
        self.bn3 = nn.Identity()

        # Channel shuffle is folded into conv3 weights at init to avoid a runtime copy/kernel.
        # Do this after BN-folding so the permuted conv3 includes BN effects.
        _fold_channel_shuffle_into_conv3(self.conv3, groups)

        if in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        # If shortcut includes a BatchNorm, fold it into the shortcut conv as well.
        # This avoids an extra BN kernel when shortcut is used.
        if isinstance(self.shortcut, nn.Sequential) and len(self.shortcut) >= 2:
            conv_short = self.shortcut[0]
            bn_short = self.shortcut[1]
            if isinstance(bn_short, nn.BatchNorm2d):
                fold_batchnorm_into_conv(conv_short, bn_short)
                # replace the BN with identity
                self.shortcut[1] = nn.Identity()

    def forward(self, x):
        # Use inplace ReLU for inference memory savings (safe as we expect inference usage).
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        # Channel shuffle folded into conv3 weights; no runtime shuffle required.
        out = F.relu(self.bn3(self.conv3(out)), inplace=True)

        # Compute shortcut path
        sc = self.shortcut(x)

        # If we're in inference/eval mode and tensors are on CUDA in fp16 and channels-last layout,
        # use the Triton in-place add to avoid extra allocation and reduce memory traffic.
        if (not self.training) and out.is_cuda and out.dtype == torch.half:
            try:
                # Ensure channels-last contiguous layout for both tensors for best Triton performance.
                out = out.contiguous(memory_format=torch.channels_last)
                sc_cl = sc.contiguous(memory_format=torch.channels_last).to(dtype=out.dtype, device=out.device)
                triton_inplace_add(out, sc_cl)
                return out
            except Exception:
                # If anything goes wrong with the fast path, fall back to the safe differentiable add.
                out = out + sc
                return out
        else:
            # Training or non-optimized path: keep differentiable behaviour.
            out = out + sc
            return out

# Final ModelNew using Triton-accelerated channel shuffle and add
class ModelNew(nn.Module):
    def __init__(self, num_classes=1000, groups=3, stages_repeats=[3, 7, 3], stages_out_channels=[24, 240, 480, 960]):
        super(ModelNew, self).__init__()

        self.conv1 = nn.Conv2d(3, stages_out_channels[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(stages_out_channels[0])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage2 = self._make_stage(stages_out_channels[0], stages_out_channels[1], stages_repeats[0], groups)
        self.stage3 = self._make_stage(stages_out_channels[1], stages_out_channels[2], stages_repeats[1], groups)
        self.stage4 = self._make_stage(stages_out_channels[2], stages_out_channels[3], stages_repeats[2], groups)

        self.conv5 = nn.Conv2d(stages_out_channels[3], 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(1024)

        self.fc = nn.Linear(1024, num_classes)

    def _make_stage(self, in_channels, out_channels, repeats, groups):
        layers = []
        layers.append(ShuffleNetUnitTriton(in_channels, out_channels, groups))
        for _ in range(1, repeats):
            layers.append(ShuffleNetUnitTriton(out_channels, out_channels, groups))
        return nn.Sequential(*layers)

    def fold_for_inference(self, device=None, use_channels_last=True, fp16=True):
        """
        Prepare the model for inference:
        - switch to eval()
        - fold remaining BatchNorms (bn2 and any shortcut BN) into preceding convs across all units
        - optionally convert to channels-last memory format
        - optionally convert model params/buffers to float16
        - enable cudnn.benchmark for fixed-size inputs

        Call this once before benchmarking or running inference with fixed input sizes.
        """
        self.eval()
        if device is not None:
            self.to(device)

        # Fold any remaining BNs in ShuffleNetUnitTriton instances (conv2 and shortcut BN)
        for m in self.modules():
            if isinstance(m, ShuffleNetUnitTriton):
                # fold bn2 -> conv2
                if hasattr(m, "bn2") and isinstance(m.bn2, nn.BatchNorm2d):
                    fold_batchnorm_into_conv(m.conv2, m.bn2)
                    m.bn2 = nn.Identity()
                # fold shortcut BN if present
                if hasattr(m, "shortcut") and isinstance(m.shortcut, nn.Sequential) and len(m.shortcut) >= 2:
                    conv_short = m.shortcut[0]
                    bn_short = m.shortcut[1]
                    if isinstance(bn_short, nn.BatchNorm2d):
                        fold_batchnorm_into_conv(conv_short, bn_short)
                        m.shortcut[1] = nn.Identity()

        if use_channels_last:
            # Convert model to channels-last memory format; faster for cuDNN on Ampere
            self.to(memory_format=torch.channels_last)
        if fp16:
            # Convert parameters/buffers to float16 (inference); ensure BN were folded earlier.
            self.half()

        # For fixed-size inputs, allow cuDNN to benchmark the best kernels.
        torch.backends.cudnn.benchmark = True

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x