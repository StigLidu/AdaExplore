import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl

# Autotune configurations for the Triton kernel
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 64},  num_warps=1, num_stages=2),
    triton.Config({"BLOCK_SIZE": 128}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=8, num_stages=2),
]


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['L'])
@triton.jit
def _fused_bn_relu_kernel(
    x_ptr,         # pointer to x_flat (C, L) flattened as C*L
    out_ptr,       # pointer to out_flat (C, L)
    weight_ptr,    # pointer to weight (C,)
    bias_ptr,      # pointer to bias (C,)
    mean_ptr,      # pointer to running_mean (C,)
    var_ptr,       # pointer to running_var (C,)
    C,             # number of channels (rows)
    L,             # length per channel (columns)
    eps,           # epsilon for batchnorm
    BLOCK_SIZE: tl.constexpr,
):
    """
    Each program handles a block of size BLOCK_SIZE of one channel (row).
    Program id 0 -> channel index
    Program id 1 -> block index along the L dimension
    """
    c = tl.program_id(0)
    block_id = tl.program_id(1)
    block_start = block_id * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < L

    row_ptr = x_ptr + c * L
    row_out_ptr = out_ptr + c * L

    x = tl.load(row_ptr + offs, mask=mask, other=0.0)

    w = tl.load(weight_ptr + c)
    b = tl.load(bias_ptr + c)
    m = tl.load(mean_ptr + c)
    v = tl.load(var_ptr + c)

    # compute scale and bias for affine batchnorm: y = (x - m) / sqrt(v+eps) * w + b
    scale = w / tl.sqrt(v + eps)
    bias_term = b - m * scale

    out = x * scale + bias_term
    # ReLU
    out = tl.where(out > 0.0, out, 0.0)

    tl.store(row_out_ptr + offs, out, mask=mask)


def fused_bn_relu_triton(x: torch.Tensor, bn: nn.BatchNorm2d):
    """
    x: (N, C, H, W) contiguous
    bn: nn.BatchNorm2d module (with running statistics and affine parameters)
    This function only uses the fused Triton kernel in inference (bn.training == False).
    In training mode we fall back to the standard PyTorch ops to preserve running-statistics updates.
    """
    assert x.is_cuda, "fused_bn_relu_triton: input must be CUDA tensor"

    N, C, H, W = x.shape
    L = N * H * W

    # Permute to (C, N, H, W) and flatten to (C, L) so each channel is contiguous
    x_perm = x.permute(1, 0, 2, 3).contiguous()
    x_flat = x_perm.view(C, L)

    # Prepare parameter tensors (ensure on same device and contiguous)
    if bn.affine:
        weight = bn.weight.contiguous().to(x.device)
        bias = bn.bias.contiguous().to(x.device)
    else:
        weight = torch.ones(C, device=x.device, dtype=x.dtype)
        bias = torch.zeros(C, device=x.device, dtype=x.dtype)

    mean = bn.running_mean.contiguous().to(x.device)
    var = bn.running_var.contiguous().to(x.device)
    eps = float(bn.eps)

    out_flat = torch.empty_like(x_flat)

    # grid: (C, num_blocks)
    def grid(meta):
        return (C, (L + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'])

    _fused_bn_relu_kernel[grid](
        x_flat,
        out_flat,
        weight,
        bias,
        mean,
        var,
        C,
        L,
        eps,
    )

    # reshape back to (N, C, H, W)
    out = out_flat.view(C, N, H, W).permute(1, 0, 2, 3).contiguous()
    return out


class FusedBatchNormReLU(nn.Module):
    """
    Wraps an nn.BatchNorm2d and fuses BatchNorm (inference) + ReLU using a Triton kernel.
    In training mode (bn.training == True) we fall back to bn(x) followed by F.relu to preserve running stats updates.
    """

    def __init__(self, bn: nn.BatchNorm2d):
        super(FusedBatchNormReLU, self).__init__()
        if not isinstance(bn, nn.BatchNorm2d):
            raise TypeError("FusedBatchNormReLU requires an nn.BatchNorm2d instance")
        self.bn = bn

    def forward(self, x: torch.Tensor):
        # Use fused Triton kernel only in inference and on CUDA tensors
        if (not self.bn.training) and x.is_cuda and x.dtype == torch.float32:
            return fused_bn_relu_triton(x, self.bn)
        else:
            # fall back to standard batchnorm + relu (keeps training semantics)
            return F.relu(self.bn(x), inplace=True)


class DenseBlock(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        """
        DenseBlock with fused BatchNorm+ReLU modules (inference fused via Triton).
        """
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(self._make_layer(num_input_features + i * growth_rate, growth_rate))
        self.layers = nn.ModuleList(layers)

    def _make_layer(self, in_features: int, growth_rate: int):
        """
        Creates a single layer with FusedBatchNormReLU, Conv2D, and Dropout.
        """
        bn = nn.BatchNorm2d(in_features)
        return nn.Sequential(
            FusedBatchNormReLU(bn),
            nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.Dropout(0.0)
        )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(x)
            features.append(new_feature)
            x = torch.cat(features, 1)  # Concatenate along channel axis
        return x


class TransitionLayer(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        """
        Transition layer with fused BatchNorm+ReLU before Conv and AvgPool.
        """
        super(TransitionLayer, self).__init__()
        bn = nn.BatchNorm2d(num_input_features)
        self.transition = nn.Sequential(
            FusedBatchNormReLU(bn),
            nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.transition(x)


class ModelNew(nn.Module):
    def __init__(self, growth_rate: int = 32, num_classes: int = 1000):
        """
        DenseNet-like model where BatchNorm+ReLU sequences are replaced with FusedBatchNormReLU,
        which uses a Triton kernel for the inference path for improved throughput on CUDA.
        """
        super(ModelNew, self).__init__()

        # Initial convolution and pooling
        # Replace the BN+ReLU pair with a fused module
        initial_bn = nn.BatchNorm2d(64)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            FusedBatchNormReLU(initial_bn),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Each dense block is followed by a transition layer, except the last one
        num_features = 64
        block_layers = [6, 12, 24, 16]  # Corresponding layers in DenseNet121

        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()

        for i, num_layers in enumerate(block_layers):
            block = DenseBlock(num_layers=num_layers, num_input_features=num_features, growth_rate=growth_rate)
            self.dense_blocks.append(block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_layers) - 1:
                transition = TransitionLayer(num_input_features=num_features, num_output_features=num_features // 2)
                self.transition_layers.append(transition)
                num_features = num_features // 2

        # Final batch norm (fused with ReLU) and classifier
        final_bn = nn.BatchNorm2d(num_features)
        self.final_bn_relu = FusedBatchNormReLU(final_bn)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)

        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transition_layers[i](x)

        # Use fused BN+ReLU here as well (it already includes ReLU)
        x = self.final_bn_relu(x)

        # Global average pooling and classifier
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.classifier(x)
        return x