import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configurations tuned for NVIDIA A6000 (Ampere)
AUTOTUNE_ADD_RELU = [
    triton.Config({"BLOCK_SIZE": 256},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 512},  num_warps=8, num_stages=2),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=3),
]

@triton.autotune(configs=AUTOTUNE_ADD_RELU, key=['n_elements'])
@triton.jit
def _add_relu_inplace_kernel(x_ptr, add_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    In-place elementwise add followed by ReLU:
      x[:] = max(x + add, 0)
    Works on flattened (1-D) tensors.
    Each program handles a contiguous block of size BLOCK_SIZE.
    """
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    a = tl.load(add_ptr + offs, mask=mask, other=0.0)
    y = x + a
    y = tl.maximum(y, 0.0)
    tl.store(x_ptr + offs, y, mask=mask)


def triton_add_relu_inplace(x: torch.Tensor, add: torch.Tensor):
    """
    Wrapper to perform in-place x = relu(x + add) using Triton kernel.
    Both x and add must be CUDA tensors with same shape and contiguous.
    """
    assert x.is_cuda and add.is_cuda, "Triton kernel requires CUDA tensors."
    # Ensure contiguous layout and same dtype/device
    x = x.contiguous()
    add = add.contiguous()
    assert x.numel() == add.numel(), "Shapes must match for add+relu."
    n = x.numel()
    grid = lambda meta: ((n + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _add_relu_inplace_kernel[grid](x.view(-1), add.view(-1), n)
    return x


def _fold_bn_into_conv(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    """
    Fold affine BatchNorm parameters into a preceding Conv2d (in-place).
    After folding, conv.bias will be created/updated and bn is expected to be replaced
    by the caller (e.g., with nn.Identity).
    """
    with torch.no_grad():
        if conv is None or bn is None:
            return

        # If bn is already Identity, nothing to do.
        if isinstance(bn, nn.Identity):
            return

        w = conv.weight.data
        device = w.device
        dtype = w.dtype

        # BN params (handle None cases)
        if getattr(bn, "weight", None) is not None:
            gamma = bn.weight.detach().to(device=device, dtype=dtype)
        else:
            gamma = torch.ones(w.shape[0], device=device, dtype=dtype)

        if getattr(bn, "bias", None) is not None:
            beta = bn.bias.detach().to(device=device, dtype=dtype)
        else:
            beta = torch.zeros(w.shape[0], device=device, dtype=dtype)

        running_mean = bn.running_mean.detach().to(device=device, dtype=dtype)
        running_var = bn.running_var.detach().to(device=device, dtype=dtype)
        eps = float(bn.eps)

        # compute scale and bias_folded
        scale = gamma / torch.sqrt(running_var + eps)
        bias_folded = beta - running_mean * scale

        # Apply scale to conv weights per out-channel
        # w: (out_chan, in_chan, kH, kW)
        w.mul_(scale.view(-1, 1, 1, 1))

        # Handle conv bias
        if conv.bias is None:
            conv.bias = nn.Parameter(bias_folded.clone())
        else:
            conv.bias.data = conv.bias.data * scale + bias_folded


class BasicBlockNew(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        BasicBlock with support for folding BatchNorm into Conv for inference.
        Uses Triton in-place add+ReLU for the residual addition when BN is folded.
        """
        super(BasicBlockNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
        self._folded = False  # whether this block's BNs have been folded into convs

    def fold(self):
        """
        Fold bn1 into conv1, bn2 into conv2, and downsample's bn into its conv (if present).
        Replace bn modules with nn.Identity to skip them in forward.
        Should be called only when in eval mode and tensors are on the target device.
        """
        if self._folded:
            return

        # fold bn1 -> conv1
        if isinstance(self.bn1, nn.BatchNorm2d):
            _fold_bn_into_conv(self.conv1, self.bn1)
            self.bn1 = nn.Identity()

        # fold bn2 -> conv2
        if isinstance(self.bn2, nn.BatchNorm2d):
            _fold_bn_into_conv(self.conv2, self.bn2)
            self.bn2 = nn.Identity()

        # fold downsample if it's a Sequential(conv, bn) pattern
        if isinstance(self.downsample, nn.Sequential):
            if len(self.downsample) >= 2 and isinstance(self.downsample[0], nn.Conv2d) and isinstance(self.downsample[1], nn.BatchNorm2d):
                conv_ds = self.downsample[0]
                bn_ds = self.downsample[1]
                _fold_bn_into_conv(conv_ds, bn_ds)
                # replace the bn in the sequential with Identity (and keep other modules intact)
                new_seq = []
                for i, m in enumerate(self.downsample):
                    if i == 1:
                        new_seq.append(nn.Identity())
                    else:
                        new_seq.append(m)
                self.downsample = nn.Sequential(*new_seq)

        self._folded = True

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        # If bn1 was folded, it's Identity and conv1 already includes BN affine.
        if isinstance(self.bn1, nn.Identity):
            out = torch.relu(out)
        else:
            out = self.bn1(out)
            out = torch.relu(out)

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # If bn2 was folded, conv2 already includes BN affine and bn2 is Identity.
        if isinstance(self.bn2, nn.Identity) and out.is_cuda and identity.is_cuda:
            # Use Triton kernel to perform in-place add + relu to avoid extra temporaries and kernel launches.
            # Ensure contiguity for Triton kernel.
            out = out.contiguous()
            identity = identity.contiguous()
            triton_add_relu_inplace(out, identity)
            return out
        else:
            if not isinstance(self.bn2, nn.Identity):
                out = self.bn2(out)
            out = out + identity
            out = torch.relu(out)
            return out


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        ResNet-like model optimized for inference on CUDA by folding BatchNorms into Convs
        and using a Triton in-place add+ReLU kernel for the residual path.

        Folding is applied lazily on first forward in eval mode on CUDA to preserve training behavior.
        """
        super(ModelNew, self).__init__()
        self.in_channels = 64

        # Initial layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(BasicBlockNew, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlockNew, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlockNew, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlockNew, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlockNew.expansion, num_classes)

        self._bns_folded = False  # whether global folding has been applied

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

    def _fold_all_bns(self):
        """
        Fold BatchNorm layers into preceding Convs across the whole network.
        This replaces BN modules with Identity (so they are skipped) and modifies Conv weights/bias.
        Only call when in eval mode and on CUDA to ensure running stats are stable and tensors on device.
        """
        if self._bns_folded:
            return

        # Fold bn1 into conv1 if applicable
        if isinstance(self.bn1, nn.BatchNorm2d):
            _fold_bn_into_conv(self.conv1, self.bn1)
            self.bn1 = nn.Identity()

        # Fold BNs inside each BasicBlockNew
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                if isinstance(block, BasicBlockNew):
                    block.fold()

        self._bns_folded = True

    def forward(self, x):
        # On CUDA in eval mode, fold all BatchNorms into Convs once to remove BN kernels.
        if x.is_cuda and (not self.training):
            if not self._bns_folded:
                # ensure parameters are on the same device as input before folding
                # This moves folded conv.bias and modified weights are in-place; do it under no_grad.
                self.to(x.device)
                self._fold_all_bns()

            # Now forward without explicit BN calls (they were folded into Convs)
            x = self.conv1(x)
            # After folding bn1 into conv1, apply ReLU (conv1 includes BN affine)
            x = torch.relu(x)

            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x
        else:
            # Standard path (training or CPU). Preserve original behavior.
            x = self.conv1(x)
            x = self.bn1(x)
            x = torch.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x