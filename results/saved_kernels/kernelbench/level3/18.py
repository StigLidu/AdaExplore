import torch
import torch.nn as nn
import torch.nn.functional as F

# Favor fast convolution algorithms for repeated sizes on CUDA
torch.backends.cudnn.benchmark = True
try:
    # Allow TF32 on Ampere to accelerate convolutions/matrix multiplies where appropriate
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
except Exception:
    pass


class Conv2dCached(nn.Module):
    """
    Wrapper around nn.Conv2d that keeps a cached copy of the weight/bias
    on a target device/dtype to avoid per-forward casting overhead.
    The original nn.Conv2d parameters are still kept as trainable parameters.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        # cached buffers for fast forward (empty until prepared)
        self.register_buffer("_cached_w", torch.empty(0))
        self.register_buffer("_cached_b", torch.empty(0))
        self._prepared = False

    def forward(self, x):
        # Use cached weights if prepared and matching device/dtype
        if self._cached_w.numel() != 0 and self._cached_w.device == x.device and self._cached_w.dtype == x.dtype:
            return F.conv2d(x, self._cached_w, None if self._cached_b.numel() == 0 else self._cached_b,
                            stride=self.conv.stride, padding=self.conv.padding,
                            dilation=self.conv.dilation, groups=self.conv.groups)
        else:
            return self.conv(x)

    def prepare(self, device=None, dtype=None):
        """
        Prepare cached weight/bias on the given device/dtype. If None, infer from parameters.
        Prefer fp16 on CUDA.
        """
        if device is None:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device('cpu')
        if dtype is None:
            dtype = torch.float16 if device.type == 'cuda' else torch.float32

        w = self.conv.weight.to(device=device, dtype=dtype).contiguous()
        if self.conv.bias is not None:
            b = self.conv.bias.to(device=device, dtype=dtype).contiguous()
        else:
            b = torch.empty(0, device=device, dtype=dtype)

        # Reuse existing buffers if shapes match to avoid allocations
        if getattr(self, "_cached_w", None) is not None and self._cached_w.numel() != 0 and self._cached_w.shape == w.shape and self._cached_w.device == device:
            self._cached_w.copy_(w)
            if self._cached_b.numel() != 0 and self._cached_b.shape == b.shape and self._cached_b.device == device:
                self._cached_b.copy_(b)
            else:
                self._cached_b = b
        else:
            self._cached_w = w
            self._cached_b = b

        self._prepared = True

    def to(self, *args, **kwargs):
        module = super(Conv2dCached, self).to(*args, **kwargs)
        # Attempt to prepare cached buffers on move
        device = None
        for p in self.parameters():
            device = p.device
            break
        dtype = kwargs.get('dtype', None)
        if dtype is None:
            dtype = torch.float16 if device is not None and device.type == 'cuda' else None
        try:
            self.prepare(device=device, dtype=dtype)
        except Exception:
            # If prepare fails now, it will be attempted lazily in forward
            pass
        return module


class FireModuleNew(nn.Module):
    """
    Fire module optimized for inference throughput:
      - Caches squeeze (1x1) weights and a fused expand (3x3) weight which embeds the 1x1 expand filters
      - Prepares cached buffers on the desired device/dtype (prefer fp16 on CUDA)
      - Performs the squeeze conv using cached weights, then a single fused expand conv (3x3, padding=1)
      - Keeps outputs in channels_last on CUDA when possible (ModelNew handles layout)
    """
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(FireModuleNew, self).__init__()

        # Keep original parameterized modules for serialization/training semantics
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1, bias=True)
        self.squeeze_activation = nn.ReLU(inplace=True)

        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1, bias=True)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1, bias=True)
        self.expand_activation = nn.ReLU(inplace=True)

        # cached buffers for fast forward (built on prepare)
        self.register_buffer('_squeeze_w', torch.empty(0))
        self.register_buffer('_squeeze_b', torch.empty(0))
        self.register_buffer('_fused_expand_w', torch.empty(0))
        self.register_buffer('_fused_expand_b', torch.empty(0))

        # track parameter versions to avoid unnecessary rebuilds
        self._v_s_w = None
        self._v_s_b = None
        self._v_e1_w = None
        self._v_e1_b = None
        self._v_e3_w = None
        self._v_e3_b = None

        self._prepared = False

    def _needs_rebuild(self):
        if self._v_s_w != getattr(self.squeeze.weight, "_version", None):
            return True
        if self._v_s_b != getattr(self.squeeze.bias, "_version", None):
            return True
        if self._v_e1_w != getattr(self.expand1x1.weight, "_version", None):
            return True
        if self._v_e1_b != getattr(self.expand1x1.bias, "_version", None):
            return True
        if self._v_e3_w != getattr(self.expand3x3.weight, "_version", None):
            return True
        if self._v_e3_b != getattr(self.expand3x3.bias, "_version", None):
            return True
        return False

    def _rebuild_cached_params(self, device, dtype):
        """
        Build cached tensors on the specified device/dtype:
          - _squeeze_w: (squeeze_channels, in_channels, 1, 1)
          - _squeeze_b: (squeeze_channels,)
          - _fused_expand_w: (expand1x1_out + expand3x3_out, squeeze_channels, 3, 3)
          - _fused_expand_b: (expand1x1_out + expand3x3_out,)
        The fused expand places the 1x1 filters' values at center [1,1] of 3x3 kernels.
        """
        out1 = self.expand1x1.out_channels
        out3 = self.expand3x3.out_channels
        s_ch = self.squeeze.out_channels

        # Move weights/biases to target device/dtype
        w_e1 = self.expand1x1.weight.to(device=device, dtype=dtype)
        b_e1 = self.expand1x1.bias.to(device=device, dtype=dtype) if self.expand1x1.bias is not None else torch.zeros((out1,), device=device, dtype=dtype)
        w_e3 = self.expand3x3.weight.to(device=device, dtype=dtype)
        b_e3 = self.expand3x3.bias.to(device=device, dtype=dtype) if self.expand3x3.bias is not None else torch.zeros((out3,), device=device, dtype=dtype)

        # Build fused expand 3x3 weights: center for expand1x1, direct copy for expand3x3
        fused_w = torch.zeros((out1 + out3, s_ch, 3, 3), device=device, dtype=dtype)
        fused_w[:out1, :, 1, 1] = w_e1.view(out1, s_ch)
        fused_w[out1:out1 + out3, :, :, :] = w_e3
        fused_b = torch.cat([b_e1, b_e3], dim=0)

        # Prepare squeeze weights/bias in target dtype/device
        s_w = self.squeeze.weight.to(device=device, dtype=dtype)
        s_b = self.squeeze.bias.to(device=device, dtype=dtype) if self.squeeze.bias is not None else torch.zeros((s_ch,), device=device, dtype=dtype)

        # Ensure contiguous for efficient conv usage
        fused_w = fused_w.contiguous()
        fused_b = fused_b.contiguous()
        s_w = s_w.contiguous()
        s_b = s_b.contiguous()

        # Reuse existing buffers where possible
        if getattr(self, '_fused_expand_w', None) is not None and self._fused_expand_w.numel() != 0 and self._fused_expand_w.shape == fused_w.shape and self._fused_expand_w.device == device:
            self._fused_expand_w.copy_(fused_w)
            if getattr(self, '_fused_expand_b', None) is not None and self._fused_expand_b.shape == fused_b.shape and self._fused_expand_b.device == device:
                self._fused_expand_b.copy_(fused_b)
            else:
                self._fused_expand_b = fused_b
        else:
            self._fused_expand_w = fused_w
            self._fused_expand_b = fused_b

        if getattr(self, '_squeeze_w', None) is not None and self._squeeze_w.numel() != 0 and self._squeeze_w.shape == s_w.shape and self._squeeze_w.device == device:
            self._squeeze_w.copy_(s_w)
            if getattr(self, '_squeeze_b', None) is not None and self._squeeze_b.shape == s_b.shape and self._squeeze_b.device == device:
                self._squeeze_b.copy_(s_b)
            else:
                self._squeeze_b = s_b
        else:
            self._squeeze_w = s_w
            self._squeeze_b = s_b

        # Update versions
        self._v_s_w = getattr(self.squeeze.weight, "_version", None)
        self._v_s_b = getattr(self.squeeze.bias, "_version", None)
        self._v_e1_w = getattr(self.expand1x1.weight, "_version", None)
        self._v_e1_b = getattr(self.expand1x1.bias, "_version", None)
        self._v_e3_w = getattr(self.expand3x3.weight, "_version", None)
        self._v_e3_b = getattr(self.expand3x3.bias, "_version", None)

    def prepare(self, device=None, dtype=None):
        """
        Prepare cached fused buffers on the given device/dtype. If None, infer device from parameters.
        Prefer fp16 on CUDA.
        """
        if device is None:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device('cpu')
        if dtype is None:
            dtype = torch.float16 if device.type == 'cuda' else torch.float32

        if (self._fused_expand_w.numel() == 0) or self._needs_rebuild() or self._fused_expand_w.device != device or self._fused_expand_w.dtype != dtype:
            self._rebuild_cached_params(device=device, dtype=dtype)
        self._prepared = True

    def to(self, *args, **kwargs):
        module = super(FireModuleNew, self).to(*args, **kwargs)
        # Attempt to prepare cached buffers on move
        device = None
        for p in self.parameters():
            device = p.device
            break
        dtype = kwargs.get('dtype', None)
        if dtype is None:
            dtype = torch.float16 if device is not None and device.type == 'cuda' else None
        try:
            self.prepare(device=device, dtype=dtype)
        except Exception:
            pass
        return module

    def forward(self, x):
        """
        Forward:
          - Squeeze: cached 1x1 conv
          - ReLU
          - Expand: cached fused 3x3 conv (1x1 expand embedded)
        """
        device = x.device
        target_dtype = torch.float16 if x.is_cuda else x.dtype

        # Ensure cached buffers prepared for current device/dtype
        if not self._prepared or self._fused_expand_w.numel() == 0 or self._fused_expand_w.device != device or self._fused_expand_w.dtype != target_dtype or self._needs_rebuild():
            # lazy prepare
            self.prepare(device=device, dtype=target_dtype)

        # Use cached squeeze weights for 1x1 conv
        x_s = F.conv2d(x, self._squeeze_w, self._squeeze_b, stride=1, padding=0)

        # Activation
        x_s = self.squeeze_activation(x_s)

        # Fused expand conv (3x3)
        out = F.conv2d(x_s, self._fused_expand_w, self._fused_expand_b, padding=1)

        out = self.expand_activation(out)
        return out


class ModelNew(nn.Module):
    """
    Optimized Model:
      - Replaces initial and classifier convolutions with Conv2dCached to avoid per-forward casts.
      - Uses FireModuleNew which caches squeeze and fused expand convs in target device/dtype.
      - Prefers channels_last layout and runs the entire forward under a single autocast FP16 scope on CUDA.
      - Prepares cached buffers when moved to a device via .to(...)
    """
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()

        # Replace first 7x7 conv and final 1x1 conv with Conv2dCached for cached fp16 buffers
        self.features = nn.Sequential(
            Conv2dCached(3, 96, kernel_size=7, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModuleNew(96, 16, 64, 64),
            FireModuleNew(128, 16, 64, 64),
            FireModuleNew(128, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModuleNew(256, 32, 128, 128),
            FireModuleNew(256, 48, 192, 192),
            FireModuleNew(384, 48, 192, 192),
            FireModuleNew(384, 64, 256, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModuleNew(512, 64, 256, 256),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.0),
            Conv2dCached(512, num_classes, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def to(self, *args, **kwargs):
        """
        Move model to device/dtype and trigger prepare on cached convs and FireModuleNew children
        so that fused buffers are available before first forward.
        """
        module = super(ModelNew, self).to(*args, **kwargs)

        # infer device/dtype for buffer preparation
        device = None
        for p in self.parameters():
            device = p.device
            break
        dtype = kwargs.get('dtype', None)
        if dtype is None and device is not None:
            dtype = torch.float16 if device.type == 'cuda' else None

        # Prepare Conv2dCached and FireModuleNew modules
        for m in self.modules():
            try:
                if isinstance(m, Conv2dCached):
                    m.prepare(device=device, dtype=dtype)
                if isinstance(m, FireModuleNew):
                    m.prepare(device=device, dtype=dtype)
            except Exception:
                # If prepare fails now, lazy prepare in forward will attempt again
                pass

        return module

    def forward(self, x):
        orig_dtype = x.dtype
        if x.is_cuda:
            # Convert to channels_last once for the entire forward pass to leverage Tensor Cores / NHWC-friendly kernels
            x = x.contiguous(memory_format=torch.channels_last)
            # Run entire forward under autocast fp16 to maximize throughput. Cached buffers were prepared in fp16.
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                x = self.features(x)
                x = self.classifier(x)
            # Convert output back to original dtype (typically fp32) for compatibility
            if x.dtype != orig_dtype:
                x = x.to(orig_dtype)
            return torch.flatten(x, 1)
        else:
            # CPU path: standard execution in fp32
            x = x.contiguous()
            x = self.features(x)
            x = self.classifier(x)
            return torch.flatten(x, 1)