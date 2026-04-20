import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class Model(nn.Module):
    """
    Complex model that demonstrates a 3D circular padding followed by a
    merge-of-depth-and-channels pattern into 2D convolution, sequence padding
    via ZeroPad1d, and a final 1x1 convolution projection.

    The computation graph:
      - Input: (N, C, D, H, W)
      - CircularPad3d -> (N, C, D_p, H_p, W_p)
      - Merge channel and depth -> (N, C * D_p, H_p, W_p)
      - Conv2d (kernel, stride) -> (N, conv_out, H_out, W_out)
      - Flatten spatial -> (N, conv_out, L = H_out * W_out)
      - ZeroPad1d (pad_left, pad_right) -> (N, conv_out, L_new)
      - Reshape back to 2D grid (N, conv_out, H_new, W_out) where H_new = L_new // W_out
      - ReLU
      - 1x1 Conv2d projection to final channels
    """
    def __init__(
        self,
        in_channels: int,
        depth: int,
        pad_d_front: int,
        pad_d_back: int,
        pad_h_top: int,
        pad_h_bottom: int,
        pad_w_left: int,
        pad_w_right: int,
        conv_out_channels: int,
        final_channels: int,
        conv_kernel_size: int,
        conv_stride: int,
        pad1_left: int,
        pad1_right: int,
        input_h: int,
        input_w: int,
    ):
        """
        Initializes the Model.

        Args:
            in_channels (int): Number of input channels.
            depth (int): Depth (D) dimension of the input.
            pad_d_front (int): Circular pad at front of depth axis.
            pad_d_back (int): Circular pad at back of depth axis.
            pad_h_top (int): Circular pad top for height.
            pad_h_bottom (int): Circular pad bottom for height.
            pad_w_left (int): Circular pad left for width.
            pad_w_right (int): Circular pad right for width.
            conv_out_channels (int): Output channels of the main Conv2d.
            final_channels (int): Output channels of the final 1x1 Conv2d.
            conv_kernel_size (int): Kernel size of the main Conv2d (square).
            conv_stride (int): Stride of the main Conv2d (applied to both H and W).
            pad1_left (int): Left padding for ZeroPad1d on the spatial sequence.
            pad1_right (int): Right padding for ZeroPad1d on the spatial sequence.
            input_h (int): Original input height H (used for shape reasoning).
            input_w (int): Original input width W (used for shape reasoning).
        """
        super(Model, self).__init__()

        # Save parameters for potential debugging or dynamic shape computations
        self.in_channels = in_channels
        self.depth = depth
        self.pad_d_front = pad_d_front
        self.pad_d_back = pad_d_back
        self.pad_h_top = pad_h_top
        self.pad_h_bottom = pad_h_bottom
        self.pad_w_left = pad_w_left
        self.pad_w_right = pad_w_right
        self.conv_out_channels = conv_out_channels
        self.final_channels = final_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_stride = conv_stride
        self.pad1_left = pad1_left
        self.pad1_right = pad1_right
        self.input_h = input_h
        self.input_w = input_w

        # CircularPad3d expects padding in the order:
        # (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
        self.pad3d = nn.CircularPad3d(
            (self.pad_w_left, self.pad_w_right,
             self.pad_h_top, self.pad_h_bottom,
             self.pad_d_front, self.pad_d_back)
        )

        # After circular padding, depth becomes:
        padded_depth = self.depth + self.pad_d_front + self.pad_d_back
        conv_in_channels = self.in_channels * padded_depth

        # Main Conv2d operates on merged (channel x depth) as channels dimension
        self.conv2d = nn.Conv2d(
            in_channels=conv_in_channels,
            out_channels=self.conv_out_channels,
            kernel_size=self.conv_kernel_size,
            stride=self.conv_stride,
            padding=0,  # padding already handled by CircularPad3d for spatial axes
            bias=True
        )

        # ZeroPad1d will pad along the flattened spatial sequence dimension
        self.zp1d = nn.ZeroPad1d((self.pad1_left, self.pad1_right))

        # Final 1x1 conv to project to desired output channels
        self.conv1x1 = nn.Conv2d(
            in_channels=self.conv_out_channels,
            out_channels=self.final_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (N, final_channels, H_out', W_out)
        """
        # x: (N, C, D, H, W)
        N = x.size(0)

        # 1) Circular pad in 3D
        x = self.pad3d(x)  # -> (N, C, D_p, H_p, W_p)

        # 2) Merge channel and depth: (N, C * D_p, H_p, W_p)
        # Use flatten to merge dims 1 (C) and 2 (D_p)
        x = x.flatten(1, 2)

        # 3) 2D convolution on the merged tensor
        x = self.conv2d(x)  # -> (N, conv_out_channels, H_out, W_out)
        _, C_out, H_out, W_out = x.shape

        # 4) Flatten spatial dims to sequence: (N, C_out, L)
        x = x.view(N, C_out, -1)  # L = H_out * W_out

        # 5) ZeroPad1d along the sequence dimension
        x = self.zp1d(x)  # -> (N, C_out, L_new)
        L_new = x.size(2)

        # 6) Reshape back to 2D grid.
        # We keep W_out constant (it does not change due to our choice of padding),
        # and compute H_new = L_new // W_out. This must be an integer in valid setups.
        if L_new % W_out != 0:
            # Fallback: if (rarely) the padding doesn't align, pad additional zeros at the end
            extra = W_out - (L_new % W_out)
            x = F.pad(x, (0, extra))  # pad the sequence at the right
            L_new = x.size(2)

        H_new = L_new // W_out
        x = x.view(N, C_out, H_new, W_out)

        # 7) Non-linearity
        x = F.relu(x)

        # 8) Final 1x1 projection
        x = self.conv1x1(x)  # -> (N, final_channels, H_new, W_out)

        return x

# Module-level configuration constants
batch_size = 4

# Input dimensions
in_channels = 3
depth = 5
input_h = 32
input_w = 32

# Circular padding amounts (depth, height, width)
pad_d_front = 1
pad_d_back = 1
pad_h_top = 2
pad_h_bottom = 2
pad_w_left = 2
pad_w_right = 2

# Conv2d params
conv_out_channels = 16
final_channels = 8
conv_kernel_size = 3
conv_stride = 2

# ZeroPad1d padding on the spatial sequence
# We choose left + right total such that it increases the flattened spatial length
# by a multiple of the width after Conv2d so it can be reshaped cleanly.
pad1_left = 8
pad1_right = 9

def get_inputs() -> List[torch.Tensor]:
    """
    Creates a sample input tensor with shape (batch_size, in_channels, depth, H, W).
    The values are random normal as in the examples.
    """
    x = torch.randn(batch_size, in_channels, depth, input_h, input_w)
    return [x]

def get_init_inputs() -> List:
    """
    Returns initialization inputs required to construct the Model.
    Order must match Model.__init__ signature.
    """
    return [
        in_channels,
        depth,
        pad_d_front,
        pad_d_back,
        pad_h_top,
        pad_h_bottom,
        pad_w_left,
        pad_w_right,
        conv_out_channels,
        final_channels,
        conv_kernel_size,
        conv_stride,
        pad1_left,
        pad1_right,
        input_h,
        input_w,
    ]