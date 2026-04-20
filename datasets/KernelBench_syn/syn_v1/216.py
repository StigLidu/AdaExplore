import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class Model(nn.Module):
    """
    Complex 3D feature extractor with lazy initialization and channel gating.
    Combines LazyConv3d, Threshold activation, pointwise convolutions, an SE-like
    channel gating (using AdaptiveAvgPool3d + Linear + sigmoid), and a residual
    path. Optionally wraps the main feature extractor in DistributedDataParallel
    when torch.distributed is initialized and use_ddp=True.
    """
    def __init__(self,
                 out_channels: int = 32,
                 threshold_val: float = 0.1,
                 threshold_replacement: float = 0.0,
                 conv_kernel: int = 3,
                 conv_stride: int = 2,
                 use_ddp: bool = False,
                 gating_hidden: int = 16):
        """
        Args:
            out_channels (int): number of output channels for the primary conv block.
            threshold_val (float): threshold cutoff for nn.Threshold.
            threshold_replacement (float): replacement value for elements below threshold.
            conv_kernel (int): kernel size for the main LazyConv3d.
            conv_stride (int): stride for the main LazyConv3d and residual mapping.
            use_ddp (bool): if True and torch.distributed is initialized, wrap the
                            feature extractor in DistributedDataParallel.
            gating_hidden (int): hidden units for the gating bottleneck (Linear layer).
        """
        super(Model, self).__init__()

        # Primary convolutional feature extractor (lazy init allows unknown in_channels)
        self.conv = nn.LazyConv3d(out_channels=out_channels,
                                  kernel_size=conv_kernel,
                                  stride=conv_stride,
                                  padding=conv_kernel // 2,
                                  bias=True)

        # Residual projection to match channels/spatial downsampling (lazy in_channels)
        self.res_conv = nn.LazyConv3d(out_channels=out_channels,
                                      kernel_size=1,
                                      stride=conv_stride,
                                      padding=0,
                                      bias=False)

        # Small pointwise convolution to mix channel information after thresholding
        self.pw_conv = nn.Conv3d(out_channels, out_channels, kernel_size=1, bias=True)

        # Threshold non-linearity
        self.threshold = nn.Threshold(threshold_val, threshold_replacement)

        # Adaptive pooling + gating (SE-style). Use Linear layers for gating bottleneck.
        # We'll flatten after adaptive pool to shape (batch, out_channels)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.gate_reduce = nn.Linear(out_channels, gating_hidden, bias=True)
        self.gate_expand = nn.Linear(gating_hidden, out_channels, bias=True)

        # A small final mixing conv to produce the final output
        self.final_conv = nn.Conv3d(out_channels, out_channels, kernel_size=1, bias=True)

        # Keep a sequential feature extractor for optional DDP wrapping
        self.feature_extractor = nn.Sequential(self.conv, self.threshold, self.pw_conv)

        # Optionally wrap the feature extractor with DDP if distributed is initialized
        self.use_ddp = use_ddp
        if self.use_ddp and dist.is_initialized():
            # Wrap the submodule (feature_extractor) rather than self to avoid
            # recursive wrapping during construction. Users must ensure the process
            # group and device setup are appropriate before constructing the model.
            self.feature_extractor = nn.parallel.DistributedDataParallel(self.feature_extractor)

        # Initialize gating linear layers weights for stable start
        nn.init.kaiming_uniform_(self.gate_reduce.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.gate_expand.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
         1. feature = feature_extractor(x)   # LazyConv3d initialized on first call
         2. gated = pointwise conv + SE-like gating
         3. residual = res_conv(x)           # projects input to same channel & spatial dims
         4. out = Swish(gated + residual)
        """
        # Primary feature block (this triggers lazy init of LazyConv3d modules)
        feat = self.feature_extractor(x)  # shape: (B, C_out, D', H', W')

        # SE-style channel gating
        pooled = self.pool(feat)  # (B, C_out, 1, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)  # (B, C_out)
        # bottleneck -> relu -> expand -> sigmoid
        z = F.relu(self.gate_reduce(pooled))  # (B, gating_hidden)
        z = torch.sigmoid(self.gate_expand(z)).view(pooled.size(0), pooled.size(1), 1, 1, 1)  # (B, C_out,1,1,1)

        # Apply gating (channel-wise scaling)
        gated = feat * z

        # Final mixing convolution
        mixed = self.final_conv(gated)

        # Residual projection (lazy -> will initialize consistent with input channels)
        residual = self.res_conv(x)

        # If spatial dims mismatch due to rounding, adapt via interpolation to match mixed
        if residual.shape[2:] != mixed.shape[2:]:
            # align spatial dims by trilinear interpolation
            residual = F.interpolate(residual, size=mixed.shape[2:], mode='trilinear', align_corners=False)

        out = mixed + residual

        # Final Swish activation: x * sigmoid(x)
        out = out * torch.sigmoid(out)
        return out

# Additional imports required for initializations
import math

# Module-level configuration (used by get_inputs and get_init_inputs)
batch_size = 4
in_channels = 3          # will be used to create input; Lazy conv will infer from this
depth = 32
height = 32
width = 32

out_channels = 32
threshold_val = 0.2
threshold_replacement = 0.0
conv_kernel = 3
conv_stride = 2
use_ddp = False  # Set True if torch.distributed is initialized externally and you want DDP wrapping
gating_hidden = 16

def get_inputs():
    """
    Returns a list with a single input tensor matching the expected input shape.
    The LazyConv3d modules will initialize on the first forward pass with this tensor.
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for the Model __init__ in the same order.
    """
    return [out_channels, threshold_val, threshold_replacement, conv_kernel, conv_stride, use_ddp, gating_hidden]