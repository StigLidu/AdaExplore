import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Volumetric-to-image aggregator that:
    1. Applies 3D reflection padding to an input volumetric tensor (N, C, D, H, W).
    2. Collapses the depth (D) dimension via mean to produce a 2D feature map (N, C, H', W').
    3. Mixes channels with a learnable 1x1 convolution (channel projection).
    4. Upsamples spatially with nearest-neighbor interpolation.
    5. Applies a non-linearity (ReLU) and then LogSoftmax across the channel dimension to produce
       log-probabilities per channel for each spatial location.
    """
    def __init__(self, pad: int, upsample_scale: int, in_channels: int, out_channels: int):
        """
        Args:
            pad (int): Number of voxels to reflect-pad on each side for all three spatial dims.
            upsample_scale (int): Integer scale factor for H and W upsampling (nearest).
            in_channels (int): Number of input channels (C).
            out_channels (int): Number of output channels after 1x1 convolution.
        """
        super(Model, self).__init__()
        # ReflectionPad3d expects a 6-element padding tuple:
        # (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
        pad_tuple = (pad, pad, pad, pad, pad, pad)
        self.refpad3d = nn.ReflectionPad3d(pad_tuple)

        # 1x1 Conv2d to mix/transform channels after collapsing depth dimension
        self.channel_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=True)

        # UpsamplingNearest2d to enlarge the spatial dimensions (H, W)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=upsample_scale)

        # Non-linearity and final log-probabilities
        self.relu = nn.ReLU(inplace=True)
        self.logsoftmax = nn.LogSoftmax(dim=1)  # apply over channel dimension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor with shape (N, C, D, H, W).

        Returns:
            torch.Tensor: Output tensor with shape (N, out_channels, H_out, W_out),
                          where H_out and W_out are increased by padding and upsampling.
        """
        # 1) Reflect-pad the volumetric tensor
        x_padded = self.refpad3d(x)  # shape -> (N, C, D + 2*pad, H + 2*pad, W + 2*pad)

        # 2) Collapse the depth dimension by computing mean across depth axis -> 4D tensor
        x_2d = x_padded.mean(dim=2)  # shape -> (N, C, H_padded, W_padded)

        # 3) Channel mixing with 1x1 convolution
        x_proj = self.channel_proj(x_2d)  # shape -> (N, out_channels, H_padded, W_padded)

        # 4) Spatial upsampling (nearest neighbor)
        x_up = self.upsample(x_proj)  # shape -> (N, out_channels, H_up, W_up)

        # 5) Non-linearity and final log-softmax across channels
        x_relu = self.relu(x_up)
        out = self.logsoftmax(x_relu)  # shape -> (N, out_channels, H_up, W_up)

        return out

# Configuration / default values
batch_size = 8
in_channels = 3
out_channels = 16
depth = 8
height = 32
width = 32
pad = 2
upsample_scale = 2

def get_inputs():
    """
    Produces a single volumetric input tensor:
    Shape: (batch_size, in_channels, depth, height, width)
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for Model.__init__:
    [pad, upsample_scale, in_channels, out_channels]
    """
    return [pad, upsample_scale, in_channels, out_channels]