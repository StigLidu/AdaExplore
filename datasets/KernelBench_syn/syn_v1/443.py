import torch
import torch.nn as nn
from typing import List

class Model(nn.Module):
    """
    Complex model that fuses 3D adaptive pooling with 2D convolution and a channel-wise Softmax gating.

    Computation steps:
    1. AdaptiveAvgPool3d to reduce (D, H, W) to (pool_d, pool_h, pool_w).
    2. Fuse depth into channels by reshaping (B, C, pool_d, pool_h, pool_w) -> (B, C * pool_d, pool_h, pool_w).
    3. Apply a 2D convolution over the fused representation.
    4. Compute a global channel descriptor by spatially averaging the convolution output.
    5. Apply Softmax across channels to produce gating weights.
    6. Re-scale the convolution feature maps by the per-channel gating weights and return the modulated feature maps.
    """
    def __init__(
        self,
        in_channels: int,
        pool_d: int,
        pool_h: int,
        pool_w: int,
        conv_out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ):
        """
        Initializes the model components.

        Args:
            in_channels (int): Number of input channels in the 3D input.
            pool_d (int): Output depth for AdaptiveAvgPool3d.
            pool_h (int): Output height for AdaptiveAvgPool3d.
            pool_w (int): Output width for AdaptiveAvgPool3d.
            conv_out_channels (int): Number of output channels for the Conv2d layer.
            kernel_size (int): Kernel size for Conv2d.
            stride (int): Stride for Conv2d.
            padding (int): Padding for Conv2d.
        """
        super(Model, self).__init__()
        # 3D adaptive pooling to a fixed (pool_d, pool_h, pool_w)
        self.pool3d = nn.AdaptiveAvgPool3d((pool_d, pool_h, pool_w))
        # After pooling we will fuse depth into channels: conv_in_channels = in_channels * pool_d
        conv_in_channels = in_channels * pool_d
        self.conv2d = nn.Conv2d(in_channels=conv_in_channels,
                                out_channels=conv_out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding)
        # Softmax across channel dimension to produce per-channel gating
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the fused 3D->2D pipeline.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, conv_out_channels, H_out, W_out),
                          where spatial dims depend on the pooling and convolution parameters.
        """
        # x: (B, C, D, H, W)
        pooled = self.pool3d(x)  # -> (B, C, pool_d, pool_h, pool_w)

        B, C, pool_d, pool_h, pool_w = pooled.shape

        # Fuse depth into channels to prepare for 2D convolution:
        # (B, C, pool_d, pool_h, pool_w) -> (B, C * pool_d, pool_h, pool_w)
        fused = pooled.reshape(B, C * pool_d, pool_h, pool_w)

        # Apply 2D convolution
        conv_out = self.conv2d(fused)  # -> (B, conv_out_channels, H_out, W_out)

        # Global spatial average to get channel descriptors: (B, conv_out_channels)
        channel_descriptor = conv_out.mean(dim=(2, 3))  # mean over H_out and W_out

        # Softmax across channels to obtain gating weights per sample
        gating = self.softmax(channel_descriptor)  # -> (B, conv_out_channels)

        # Reshape gating to broadcast over spatial dims and modulate conv_out
        gating = gating.view(B, gating.size(1), 1, 1)  # -> (B, conv_out_channels, 1, 1)
        out = conv_out * gating  # channel-wise gating

        return out


# Configuration / module-level variables
batch_size = 8
in_channels = 16
depth = 8
height = 64
width = 64

# Adaptive pooling targets
pool_d = 4
pool_h = 16
pool_w = 16

# Conv2d parameters
conv_out_channels = 32
kernel_size = 3
stride = 1
padding = 1

def get_inputs() -> List[torch.Tensor]:
    """
    Generates sample input tensors for the model.

    Returns:
        List[torch.Tensor]: A single-element list containing a random input tensor of shape
                            (batch_size, in_channels, depth, height, width).
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters for the Model constructor, in order.

    Returns:
        List: [in_channels, pool_d, pool_h, pool_w, conv_out_channels, kernel_size, stride, padding]
    """
    return [in_channels, pool_d, pool_h, pool_w, conv_out_channels, kernel_size, stride, padding]