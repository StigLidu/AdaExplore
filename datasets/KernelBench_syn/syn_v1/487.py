import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class Model(nn.Module):
    """
    Complex module that fuses a 3D volumetric input with a 2D feature map.
    Pipeline:
      - 3D max pooling reduces spatial resolution of volume
      - collapse depth dimension by averaging
      - 2D average pooling further reduces spatial dims
      - parallel 2D average pooling downscales the 2D feature input
      - ReLU activation on the 2D feature input
      - element-wise fusion (Hadamard product) of the two downsampled maps
      - channel mixing via a learnable channel projection matrix + bias
      - final ReLU nonlinearity
    """
    def __init__(self, channels: int):
        """
        Args:
            channels (int): Number of channels for both inputs and channel-mixing weight.
        """
        super(Model, self).__init__()
        self.channels = channels

        # 3D max pooling reduces (D,H,W) by factor 2 in each dimension
        self.maxpool3d = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # After collapsing depth, apply a 2D average pool with kernel 2 (halves H and W)
        self.avgpool_small = nn.AvgPool2d(kernel_size=2, stride=2)

        # Downsample the separate 2D feature input from original spatial size (e.g., 64->16)
        # We use kernel=4 and stride=4 to go from 64 -> 16 in the example configuration below.
        self.avgpool_down = nn.AvgPool2d(kernel_size=4, stride=4)

        # Nonlinearity
        self.relu = nn.ReLU(inplace=True)

        # Channel mixing: learnable projection matrix (out_channels x in_channels)
        # Here we keep in_channels == out_channels == channels for simplicity.
        self.weight = nn.Parameter(torch.empty(channels, channels))
        self.bias = nn.Parameter(torch.empty(channels))

        # Initialize parameters
        init.kaiming_uniform_(self.weight, a=math.sqrt(5) if 'math' in globals() else 1.0)
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
        init.uniform_(self.bias, -bound, bound)

    def forward(self, vol: torch.Tensor, feat2d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vol (torch.Tensor): 5D tensor of shape (B, C, D, H, W)
            feat2d (torch.Tensor): 4D tensor of shape (B, C, H_orig, W_orig)

        Returns:
            torch.Tensor: 4D tensor of shape (B, C, H_out, W_out) after fusion and channel mixing
        """
        # vol -> (B, C, D/2, H/2, W/2)
        pooled3 = self.maxpool3d(vol)

        # collapse depth by averaging -> (B, C, H/2, W/2)
        collapsed = pooled3.mean(dim=2)

        # further spatial reduction -> (B, C, H/4, W/4)
        pooled2 = self.avgpool_small(collapsed)

        # downsample the 2D feature map from original resolution (e.g., 64x64 -> 16x16)
        feat_down = self.avgpool_down(feat2d)

        # non-linear activation on features
        feat_activated = self.relu(feat_down)

        # element-wise fusion of the two processed maps (Hadamard product)
        fused = pooled2 * feat_activated  # shape: (B, C, H_out, W_out)

        # channel mixing using learned weight matrix W (out_c x in_c)
        # fused: (B, C_in, H, W), weight: (C_out, C_in) => result: (B, C_out, H, W)
        mixed = torch.einsum('bcih,oc->boih', fused, self.weight)

        # add bias (broadcast over spatial dims)
        mixed = mixed + self.bias.view(1, -1, 1, 1)

        # final non-linearity
        out = self.relu(mixed)
        return out

# Required for initialization above
import math

# Configuration (module-level)
batch_size = 8
channels = 32
depth = 16
height = 64
width = 64

def get_inputs():
    """
    Returns sample inputs for the Model:
      - vol: (B, C, D, H, W)
      - feat2d: (B, C, H, W)
    """
    vol = torch.randn(batch_size, channels, depth, height, width)
    feat2d = torch.randn(batch_size, channels, height, width)
    return [vol, feat2d]

def get_init_inputs():
    """
    Returns initialization arguments for Model constructors.
    Here we only need the number of channels.
    """
    return [channels]