import torch
import torch.nn as nn
from typing import Tuple

class Model(nn.Module):
    """
    Complex 3D processing model demonstrating:
      - Circular padding in 3D (nn.CircularPad3d)
      - 3D convolution + max-pooling with indices (nn.MaxPool3d)
      - Max unpooling to restore spatial resolution (nn.MaxUnpool3d)
      - Channel-wise dropout applied via nn.Dropout1d after flattening spatial dims
      - Fusion (skip connection) and final 1x1x1 convolution followed by global spatial average

    The model expects inputs of shape (N, C_in, D, H, W) and returns (N, C_out) after global pooling.
    """
    def __init__(
        self,
        pool_kernel: int,
        pool_stride: int,
        pad: Tuple[int, int, int, int, int, int],
        dropout_p: float,
        in_channels: int,
        out_channels: int
    ):
        """
        Initialize the model.

        Args:
            pool_kernel (int): Kernel size for MaxPool3d and MaxUnpool3d (assumed cubic).
            pool_stride (int): Stride for MaxPool3d/MaxUnpool3d (assumed same in all dims).
            pad (tuple): 6-tuple padding for CircularPad3d (left,right, top,bottom, front,back).
            dropout_p (float): Probability for Dropout1d.
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels after final conv.
        """
        super(Model, self).__init__()

        # Store params
        self.pool_kernel = pool_kernel
        self.pool_stride = pool_stride
        self.pad = pad
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_p = dropout_p

        # Circular padding to emulate wrap-around boundaries
        self.pad3d = nn.CircularPad3d(pad)

        # First convolution: use kernel_size=3 and no extra padding because we applied CircularPad3d
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)

        # MaxPool3d that returns indices for MaxUnpool3d
        self.pool = nn.MaxPool3d(kernel_size=pool_kernel, stride=pool_stride, return_indices=True)

        # MaxUnpool3d to invert the MaxPool3d operation (requires indices)
        self.unpool = nn.MaxUnpool3d(kernel_size=pool_kernel, stride=pool_stride, padding=0)

        # Dropout1d to zero entire channels across flattened spatial dimension.
        # We'll reshape (N, C, D', H', W') -> (N, C, L) with L = D'*H'*W' before applying dropout1d.
        self.dropout1d = nn.Dropout1d(p=dropout_p)

        # Final 1x1x1 convolution to mix channels after fusion
        self.conv2 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Sequence:
         - CircularPad3d
         - Conv3d (3x3x3) + ReLU
         - MaxPool3d (return indices)
         - Flatten spatial dims and apply Dropout1d to zero entire channels
         - MaxUnpool3d using stored indices (output_size set to conv1 output size)
         - Fuse unpooled tensor with conv1 output (skip connection)
         - 1x1x1 conv to project channels
         - Global average pooling over D,H,W to produce (N, out_channels)

        Args:
            x (torch.Tensor): Input tensor (N, C_in, D, H, W)

        Returns:
            torch.Tensor: Output tensor (N, out_channels)
        """
        # 1) Circular padding
        x_padded = self.pad3d(x)  # shape: (N, C, D+pad, H+pad, W+pad)

        # 2) Conv + ReLU. Because we padded by 1 on each side (typical), conv kernel_size=3 restores original size.
        conv_out = self.conv1(x_padded)   # shape: (N, C, D, H, W)
        conv_out = self.relu(conv_out)

        # Store conv_out size for unpooling
        conv_out_size = conv_out.size()

        # 3) MaxPool3d with indices
        pooled, indices = self.pool(conv_out)  # pooled shape: (N, C, Dp, Hp, Wp)

        # 4) Apply Dropout1d across channels by flattening spatial dims into length L
        N, C, Dp, Hp, Wp = pooled.shape
        pooled_flat = pooled.view(N, C, -1)  # (N, C, L)
        pooled_dropped = self.dropout1d(pooled_flat)  # zero entire channels with probability p
        pooled_dropped = pooled_dropped.view(N, C, Dp, Hp, Wp)  # restore shape

        # 5) MaxUnpool3d to reconstruct spatial resolution -> output_size ensures exact shape
        unpooled = self.unpool(pooled_dropped, indices, output_size=conv_out_size)  # (N, C, D, H, W)

        # 6) Fuse with skip connection (conv_out) to combine pre-pooled features
        fused = unpooled + conv_out  # element-wise addition

        # 7) Final 1x1x1 conv to mix channels and reduce to out_channels
        out = self.conv2(fused)  # (N, out_channels, D, H, W)

        # 8) Global average pooling over spatial dims
        out = out.mean(dim=[2, 3, 4])  # (N, out_channels)

        return out

# Module-level configuration variables
batch_size = 4
in_channels = 8
out_channels = 12
depth = 16
height = 16
width = 16
pool_kernel = 2
pool_stride = 2
pad_size = 1  # pad each side by 1 for each spatial dim
# CircularPad3d expects 6 values: (left, right, top, bottom, front, back)
pad_tuple = (pad_size, pad_size, pad_size, pad_size, pad_size, pad_size)
dropout_p = 0.25

def get_inputs():
    """
    Returns a list with a single input tensor of shape (batch_size, in_channels, depth, height, width).
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters for the Model constructor in the same order.
    """
    return [pool_kernel, pool_stride, pad_tuple, dropout_p, in_channels, out_channels]