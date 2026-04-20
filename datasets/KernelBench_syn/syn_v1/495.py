import torch
import torch.nn as nn
from typing import Tuple, Union

class Model(nn.Module):
    """
    Composite model that transforms a 1D signal into a 2D feature map,
    upsamples it, applies a lazily-initialized 2D convolution, non-linearity,
    and global spatial pooling to produce compact feature vectors.

    The processing pipeline is:
        1. ZeroPad1d on the input 1D tensor
        2. Unsqueeze to create a (H, W=1) 2D layout
        3. UpsamplingNearest2d to expand spatial dimensions
        4. nn.LazyConv2d to produce output channels (lazily infers in_channels)
        5. nn.ReLU activation
        6. nn.AdaptiveAvgPool2d to (1,1) followed by squeeze to (N, C)
    """
    def __init__(
        self,
        padding: int,
        upsample_scale: Union[int, Tuple[int, int]],
        conv_out_channels: int,
        conv_kernel_size: Union[int, Tuple[int, int]],
        conv_stride: int = 1,
        conv_padding: Union[int, Tuple[int, int]] = 0,
    ):
        """
        Args:
            padding (int): Amount of zero-padding applied on both sides in 1D.
            upsample_scale (int or tuple): Scale factor for (H, W) upsampling.
            conv_out_channels (int): Number of output channels for LazyConv2d.
            conv_kernel_size (int or tuple): Kernel size for LazyConv2d.
            conv_stride (int): Stride for LazyConv2d.
            conv_padding (int or tuple): Padding for LazyConv2d.
        """
        super(Model, self).__init__()
        # Pad the 1D signal before converting to 2D layout
        self.pad1d = nn.ZeroPad1d(padding)

        # Upsample the (H, W=1) layout to increase spatial size for 2D conv
        self.upsample = nn.UpsamplingNearest2d(scale_factor=upsample_scale)

        # Lazily initialized Conv2d: in_channels will be inferred at first forward
        self.conv2d = nn.LazyConv2d(
            out_channels=conv_out_channels,
            kernel_size=conv_kernel_size,
            stride=conv_stride,
            padding=conv_padding,
        )

        # Non-linearity and pooling to get fixed-size vector per sample
        self.act = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, conv_out_channels).
        """
        # 1. Pad the 1D sequence -> (N, C, Lp)
        x_padded = self.pad1d(x)

        # 2. Convert to a 2D spatial layout by adding a width dimension -> (N, C, H=Lp, W=1)
        x_2d = x_padded.unsqueeze(-1)

        # 3. Upsample spatially -> (N, C, H', W')
        x_up = self.upsample(x_2d)

        # 4. Apply lazily-initialized Conv2d -> (N, C_out, H', W')
        x_conv = self.conv2d(x_up)

        # 5. Non-linearity
        x_act = self.act(x_conv)

        # 6. Global spatial pooling to (1,1) and squeeze -> (N, C_out)
        x_pooled = self.pool(x_act).squeeze(-1).squeeze(-1)

        return x_pooled

# Configuration / hyper-parameters
batch_size = 8
in_channels = 3
input_length = 16

# Initialization parameters for the model
pad = 2  # ZeroPad1d pads both sides by this value -> length increases by 2*pad
upsample_scale = (2, 4)  # (scale_height, scale_width)
conv_out_channels = 8
conv_kernel_size = (3, 3)
conv_stride = 1
conv_padding = (1, 1)  # keep spatial dims after convolution

def get_inputs():
    """
    Returns a list with a single input tensor shaped (batch_size, in_channels, input_length).
    """
    x = torch.randn(batch_size, in_channels, input_length)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters for the Model constructor in the expected order.
    """
    return [pad, upsample_scale, conv_out_channels, conv_kernel_size, conv_stride, conv_padding]