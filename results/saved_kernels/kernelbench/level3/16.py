import torch
import torch.nn as nn
import torch.nn.functional as F
# Removed Triton-based BN+ReLU kernel and replaced it with a
# BN->Conv/Linear folding helper which is applied lazily at inference time.
# Folding BatchNorm into the preceding Conv/Linear eliminates a full-tensor
# kernel and substantially reduces memory traffic at eval time.

def _fold_bn_into_conv(bn: nn.BatchNorm2d, conv: nn.Module):
    """
    Fold BatchNorm2d 'bn' into Conv2d or Linear module 'conv'.
    This modifies conv.weight and conv.bias (creating bias if missing).
    The function uses the original conv weights (cloned) to compute the correct bias contribution.
    """
    if not isinstance(bn, nn.BatchNorm2d):
        return
    eps = bn.eps
    # handle affine vs non-affine batchnorm
    if bn.affine:
        gamma = bn.weight.data.clone()
        beta = bn.bias.data.clone()
    else:
        gamma = torch.ones(bn.num_features, device=bn.running_mean.device, dtype=bn.running_mean.dtype)
        beta = torch.zeros(bn.num_features, device=bn.running_mean.device, dtype=bn.running_mean.dtype)

    mean = bn.running_mean.data.clone()
    var = bn.running_var.data.clone()
    factor = (-gamma / torch.sqrt(var + eps) * mean + beta)  # vector of length C : (-scale*mean + beta)
    scale = gamma / torch.sqrt(var + eps)  # per-channel scale

    if isinstance(conv, nn.Conv2d):
        # conv.weight: (out, in, kh, kw)
        w_orig = conv.weight.data.clone()
        # scale conv weights per input channel (columns)
        conv.weight.data.copy_(w_orig * scale.view(1, -1, 1, 1))
        # bias contribution: sum over kernel spatial dims of original weights times factor per input
        bias_contrib = (w_orig.sum(dim=(2, 3)) * factor.view(1, -1)).sum(dim=1)  # (out,)
        if conv.bias is None:
            conv.bias = nn.Parameter(bias_contrib)
        else:
            conv.bias.data.add_(bias_contrib)
    elif isinstance(conv, nn.Linear):
        # conv.weight: (out, in)
        w_orig = conv.weight.data.clone()
        conv.weight.data.copy_(w_orig * scale.view(1, -1))
        # bias contribution: w_orig @ factor
        bias_contrib = w_orig.matmul(factor)
        if conv.bias is None:
            conv.bias = nn.Parameter(bias_contrib)
        else:
            conv.bias.data.add_(bias_contrib)
    else:
        # Unsupported module type for folding
        return


# Lazy folding helper to collect and fold all BN modules that precede conv/linear layers
def _fold_all_bns_in_model(model: nn.Module):
    # Initial conv1 & bn1 (ModelNew)
    if hasattr(model, "conv1") and hasattr(model, "bn1"):
        _fold_bn_into_conv(model.bn1, model.conv1)
        model.bn1 = nn.Identity()

    # Dense layers: DenseBlockNew -> DenseLayerNew (bn before conv)
    if hasattr(model, "dense_blocks"):
        for block in model.dense_blocks:
            # block.layers is ModuleList of DenseLayerNew
            for layer in getattr(block, "layers", []):
                if hasattr(layer, "bn") and hasattr(layer, "conv"):
                    _fold_bn_into_conv(layer.bn, layer.conv)
                    layer.bn = nn.Identity()

    # Transition layers: TransitionLayerNew (bn before conv)
    if hasattr(model, "transition_layers"):
        for t in model.transition_layers:
            if hasattr(t, "bn") and hasattr(t, "conv"):
                _fold_bn_into_conv(t.bn, t.conv)
                t.bn = nn.Identity()

    # Final batchnorm into classifier (Linear)
    if hasattr(model, "final_bn") and hasattr(model, "classifier"):
        _fold_bn_into_conv(model.final_bn, model.classifier)
        model.final_bn = nn.Identity()

    model._bn_folded = True


class DenseLayerNew(nn.Module):
    """
    Single layer inside DenseBlock but using fused batchnorm+relu Triton kernel.
    Each layer: BN -> ReLU -> Conv2d -> (Dropout omitted since p=0.0)
    """
    def __init__(self, in_features: int, growth_rate: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_features)
        self.conv = nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.bn(x)
        out = F.relu(out, inplace=True)
        out = self.conv(out)
        return out


class DenseBlockNew(nn.Module):
    """
    DenseBlock optimized to avoid repeated concatenation allocations.
    Preallocates the output tensor for the entire block and fills it slice-by-slice.
    """
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        super().__init__()
        self.num_layers = num_layers
        self.num_input_features = num_input_features
        self.growth_rate = growth_rate
        layers = []
        for i in range(num_layers):
            in_feat = num_input_features + i * growth_rate
            layers.append(DenseLayerNew(in_feat, growth_rate))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        x: (N, num_input_features, H, W)
        returns: (N, num_input_features + num_layers * growth_rate, H, W)
        """
        N, C_in, H, W = x.shape
        total_channels = C_in + self.num_layers * self.growth_rate
        device = x.device
        dtype = x.dtype

        # Preallocate output tensor and copy original input into the first channels
        out = torch.empty((N, total_channels, H, W), device=device, dtype=dtype)
        out[:, :C_in, :, :].copy_(x)

        curr_channels = C_in
        # For each layer, pass the current filled-prefix view to the layer and store the new feature
        for layer in self.layers:
            inp_view = out[:, :curr_channels, :, :]
            new_feat = layer(inp_view)
            out[:, curr_channels:curr_channels + self.growth_rate, :, :].copy_(new_feat)
            curr_channels += self.growth_rate

        return out


class TransitionLayerNew(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_input_features)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.bn(x)
        out = F.relu(out, inplace=True)
        out = self.conv(out)
        out = self.pool(out)
        return out


class ModelNew(nn.Module):
    def __init__(self, growth_rate: int = 32, num_classes: int = 1000):
        super(ModelNew, self).__init__()

        # Initial convolution and pooling
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Dense blocks and transitions
        num_features = 64
        block_layers = [6, 12, 48, 32]  # DenseNet201-like configuration

        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()

        for i, num_layers in enumerate(block_layers):
            block = DenseBlockNew(num_layers=num_layers, num_input_features=num_features, growth_rate=growth_rate)
            self.dense_blocks.append(block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_layers) - 1:
                transition = TransitionLayerNew(num_input_features=num_features, num_output_features=num_features // 2)
                self.transition_layers.append(transition)
                num_features = num_features // 2

        # Final batch norm and classifier
        self.final_bn = nn.BatchNorm2d(num_features)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If the model is in eval() mode, fold BatchNorms into Conv/Linear modules once.
        if not self.training and not getattr(self, "_bn_folded", False):
            _fold_all_bns_in_model(self)

        # Initial conv + bn + relu + maxpool
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.maxpool(x)

        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transition_layers[i](x)

        # final BN + ReLU
        x = self.final_bn(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.classifier(x)
        return x


# Compatibility helper functions (used by external test harnesses)
batch_size = 10
num_classes = 10
height, width = 224, 224  # Standard input size for DenseNet

def get_inputs():
    # Return CUDA tensors for benchmarking harnesses
    return [torch.rand(batch_size, 3, height, width).cuda()]

def get_init_inputs():
    return [32, num_classes]