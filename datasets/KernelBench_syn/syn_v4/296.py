import torch
import torch.nn as nn
from typing import List, Tuple

"""
Configuration / module-level variables:
- Defines typical sizes used to generate inputs and initialization parameters.
- conv2d_out_channels must be divisible by depth_slices for channel-to-depth reshaping.
"""
batch_size = 8
in_channels = 3
height = 128
width = 128

# Conv2d -> produces this many output channels (will be split into depth slices)
conv2d_out_channels = 32
conv2d_kernel_size = 3

# How many depth slices to create by splitting channels after Conv2d
depth_slices = 4  # must divide conv2d_out_channels

# Adaptive pooling target spatial size (H_out, W_out)
pool_output_size = (16, 16)

# Conv3d parameters
conv3d_out_channels = 16
conv3d_kernel = (3, 3, 3)  # (kD, kH, kW)


class Model(nn.Module):
    """
    Complex model combining LazyConv2d (2D convolution with lazy in_channels),
    AdaptiveAvgPool2d, channel-to-depth reshaping, and Conv3d.

    Computation pipeline:
      x (N, C_in, H, W)
        -> LazyConv2d -> ReLU (N, C2, H, W)
        -> AdaptiveAvgPool2d -> (N, C2, Hp, Wp)
        -> reshape channels into depth slices -> (N, C_group, D, Hp, Wp)
        -> Conv3d -> (N, C3_out, D, Hp, Wp)
        -> global mean over (D, Hp, Wp) -> (N, C3_out)

    Notes:
      - conv2d_out_channels must be divisible by depth_slices.
      - LazyConv2d allows in_channels to be inferred at first forward.
    """
    def __init__(
        self,
        conv2d_out_channels: int,
        depth_slices: int,
        pool_output_size: Tuple[int, int],
        conv3d_out_channels: int,
        conv3d_kernel: Tuple[int, int, int],
        conv2d_kernel: int = 3
    ):
        super(Model, self).__init__()

        if conv2d_out_channels % depth_slices != 0:
            raise ValueError("conv2d_out_channels must be divisible by depth_slices")

        self.conv2d_out_channels = conv2d_out_channels
        self.depth_slices = depth_slices
        self.group_channels = conv2d_out_channels // depth_slices

        # LazyConv2d: will infer in_channels on the first forward
        self.conv2d = nn.LazyConv2d(
            out_channels=conv2d_out_channels,
            kernel_size=conv2d_kernel,
            padding=conv2d_kernel // 2,
            bias=True
        )
        self.act = nn.ReLU(inplace=True)

        # Spatial aggregator to reduce spatial dims to a known fixed size
        self.pool = nn.AdaptiveAvgPool2d(pool_output_size)

        # Conv3d expects input shape (N, C_group, D, H, W)
        # Use padding to preserve spatial/depth dims (same padding)
        pad_d = conv3d_kernel[0] // 2
        pad_h = conv3d_kernel[1] // 2
        pad_w = conv3d_kernel[2] // 2
        self.conv3d = nn.Conv3d(
            in_channels=self.group_channels,
            out_channels=conv3d_out_channels,
            kernel_size=conv3d_kernel,
            padding=(pad_d, pad_h, pad_w),
            bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (N, C_in, H, W)

        Returns:
            Tensor of shape (N, conv3d_out_channels) containing global pooled features
            after the 3D convolutional processing.
        """
        # 1) 2D convolution (lazy in_channels inference) + non-linearity
        y = self.conv2d(x)               # (N, C2, H, W)
        y = self.act(y)

        # 2) Adaptive spatial pooling to reduce to fixed (Hp, Wp)
        y = self.pool(y)                 # (N, C2, Hp, Wp)

        N, C2, Hp, Wp = y.shape
        D = self.depth_slices
        # 3) Reshape channels into depth slices:
        #    target grouping: (N, D, C_group, Hp, Wp) -> then permute to (N, C_group, D, Hp, Wp)
        y = y.view(N, D, self.group_channels, Hp, Wp)
        y = y.permute(0, 2, 1, 3, 4).contiguous()  # (N, C_group, D, Hp, Wp)

        # 4) 3D convolution across (depth, height, width)
        y = self.conv3d(y)               # (N, C3_out, D, Hp, Wp)

        # 5) Global average pooling across spatial and depth dims -> vector
        #    Equivalent to AdaptiveAvgPool3d((1,1,1)) followed by squeeze.
        y = y.mean(dim=[2, 3, 4])        # (N, C3_out)

        return y


def get_inputs() -> List[torch.Tensor]:
    """
    Generates a single 4D input tensor for the model.

    Returns:
        list: [x] where x has shape (batch_size, in_channels, height, width)
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs() -> List:
    """
    Returns initialization parameters for Model in the same order as the constructor.

    Returns:
        list: [conv2d_out_channels, depth_slices, pool_output_size, conv3d_out_channels, conv3d_kernel, conv2d_kernel_size]
    """
    return [
        conv2d_out_channels,
        depth_slices,
        pool_output_size,
        conv3d_out_channels,
        conv3d_kernel,
        conv2d_kernel_size
    ]