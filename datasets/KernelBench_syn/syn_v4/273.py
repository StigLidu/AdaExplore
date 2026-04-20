import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    A moderately complex 2D feature processing module that combines:
    - an initial convolutional projection
    - lazy instance normalization (num_features inferred on first forward)
    - ReLU non-linearity
    - 2D average pooling to reduce spatial resolution
    - adaptive average pooling to a fixed spatial output size
    - a 1x1 projection to the desired output channels
    - a skip path created from the original input (adaptive pooled and projected)
    - residual fusion and final activation

    This structure demonstrates mixing pooling layers (AvgPool2d, AdaptiveAvgPool2d)
    with a lazy normalization layer (LazyInstanceNorm2d) and simple convolutional
    projections. The LazyInstanceNorm2d will infer its number of features from the
    first forward pass, making initialization slightly deferred.
    """
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        adaptive_output_size,  # int or tuple
        avg_pool_kernel: int = 2,
        avg_pool_stride: int = 2,
        avg_pool_padding: int = 0,
        conv_kernel: int = 3,
        conv_padding: int = 1
    ):
        super(Model, self).__init__()
        # Primary feature extractor
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=conv_kernel, padding=conv_padding)
        # Lazy instance norm will infer num_features from conv1 output on first forward
        self.inst_norm = nn.LazyInstanceNorm2d()
        self.relu = nn.ReLU(inplace=True)

        # Reduce spatial resolution with a regular average pool
        self.avg_pool = nn.AvgPool2d(kernel_size=avg_pool_kernel, stride=avg_pool_stride, padding=avg_pool_padding)

        # Reduce/reshape to a fixed spatial size regardless of input resolution
        self.adaptive_pool = nn.AdaptiveAvgPool2d(adaptive_output_size)
        self.adaptive_size = adaptive_output_size  # keep for skip path pooling

        # Project features to the desired output channel dimensionality
        self.proj = nn.Conv2d(mid_channels, out_channels, kernel_size=1)

        # A 1x1 projection for the skip path so residual addition matches channels
        self.skip_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Final non-linearity
        self.final_act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the composed operations.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, H_out, W_out)
                where (H_out, W_out) == adaptive_output_size
        """
        # Primary path
        y = self.conv1(x)               # conv -> (B, mid_channels, H, W)
        y = self.inst_norm(y)           # lazy instance norm -> (B, mid_channels, H, W)
        y = self.relu(y)                # non-linearity
        y = self.avg_pool(y)            # reduce spatial res -> (B, mid_channels, H//2, W//2) typically
        y = self.adaptive_pool(y)       # fixed spatial size -> (B, mid_channels, H_out, W_out)
        y = self.proj(y)                # project to out_channels -> (B, out_channels, H_out, W_out)

        # Skip/residual path: adaptive pooling of the original input then project
        skip = F.adaptive_avg_pool2d(x, self.adaptive_size)  # (B, in_channels, H_out, W_out)
        skip = self.skip_proj(skip)                          # (B, out_channels, H_out, W_out)

        # Residual fusion
        out = y + skip

        # Final activation
        out = self.final_act(out)

        return out

# Configuration / default parameters for test instantiation
batch_size = 8
in_channels = 3
height = 64
width = 64

mid_channels = 32
out_channels = 16
adaptive_output_size = (4, 4)

avg_pool_kernel = 2
avg_pool_stride = 2
avg_pool_padding = 0

conv_kernel = 3
conv_padding = 1

def get_inputs():
    """
    Returns the input tensors for a forward pass.
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters for the Model constructor in order.
    """
    return [
        in_channels,
        mid_channels,
        out_channels,
        adaptive_output_size,
        avg_pool_kernel,
        avg_pool_stride,
        avg_pool_padding,
        conv_kernel,
        conv_padding
    ]