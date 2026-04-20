import torch
import torch.nn as nn
from typing import Tuple, List

"""
Complex volume-to-feature reduction model that:
- Pads a 3D volumetric input (ZeroPad3d)
- Upsamples spatial dimensions with trilinear upsampling (nn.Upsample)
- Collapses the depth dimension into channels, applies 2D max pooling (nn.MaxPool2d)
- Restores the original channel/depth split and reduces across depth (mean)
This creates a non-trivial interplay between 3D and 2D ops and reshaping.
"""

# Configuration variables
batch_size = 8
channels = 4
depth = 5
height = 64
width = 48

# Initialization configuration for the Model
PAD = (1, 2, 0, 1, 1, 0)          # (left, right, top, bottom, front, back)
SCALE_FACTOR = (1.0, 2.0, 2.0)    # (depth_scale, height_scale, width_scale)
POOL_KERNEL = 3                   # kernel size for MaxPool2d

class Model(nn.Module):
    """
    Model that mixes 3D padding and upsampling with 2D pooling by
    reshaping the depth dimension into channels, pooling, and then
    reducing across depth.

    Forward pass steps:
    1. Zero-pad the input volume with ZeroPad3d.
    2. Upsample (trilinear) to increase spatial resolution (height & width).
    3. Merge depth into channel dimension to get a 4D tensor for MaxPool2d.
    4. Apply MaxPool2d to downsample spatially.
    5. Restore the (channel, depth) split and collapse depth by mean.
    """
    def __init__(self,
                 pad: Tuple[int, int, int, int, int, int],
                 scale_factor: Tuple[float, float, float],
                 pool_kernel: int):
        """
        Args:
            pad: Tuple specifying ZeroPad3d padding (left, right, top, bottom, front, back).
            scale_factor: 3-tuple for nn.Upsample scale factors (D, H, W).
            pool_kernel: kernel size for MaxPool2d.
        """
        super(Model, self).__init__()
        # ZeroPad3d pads (left, right, top, bottom, front, back)
        self.pad3d = nn.ZeroPad3d(pad)
        # Upsample in 3D using trilinear interpolation
        self.upsample3d = nn.Upsample(scale_factor=scale_factor, mode='trilinear', align_corners=False)
        # MaxPool2d expects a 4D tensor (N, C, H, W)
        # We'll use stride=2 and padding=1 to keep things nicely downsampled
        self.pool2d = nn.MaxPool2d(kernel_size=pool_kernel, stride=2, padding=1)
        # Store configuration for use in forward
        self._pad = pad
        self._scale = scale_factor
        self._pool_kernel = pool_kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, D, H, W)

        Returns:
            Tensor of shape (B, C, H_out, W_out) where depth has been reduced via mean.
        """
        # 1) Pad the 3D volume
        x = self.pad3d(x)  # shape -> (B, C, D_p, H_p, W_p)

        # 2) Upsample spatial dims (D may remain same if scale_factor[0] == 1.0)
        x = self.upsample3d(x)  # shape -> (B, C, D_u, H_u, W_u)
        B, C, D_u, H_u, W_u = x.shape

        # 3) Merge depth into channels to use 2D pooling across spatial dims
        # Resulting shape -> (B, C * D_u, H_u, W_u)
        x = x.reshape(B, C * D_u, H_u, W_u)

        # 4) Apply 2D max pooling to downsample spatial resolution
        x = self.pool2d(x)  # shape -> (B, C * D_u, H_pooled, W_pooled)
        _, _, H_pooled, W_pooled = x.shape

        # 5) Restore (C, D_u) split: shape -> (B, C, D_u, H_pooled, W_pooled)
        x = x.view(B, C, D_u, H_pooled, W_pooled)

        # 6) Reduce across depth dimension by taking the mean -> (B, C, H_pooled, W_pooled)
        x = torch.mean(x, dim=2)

        return x


def get_inputs() -> List[torch.Tensor]:
    """
    Creates a random volumetric input tensor of shape (batch_size, channels, depth, height, width).
    """
    x = torch.randn(batch_size, channels, depth, height, width, dtype=torch.float32)
    return [x]


def get_init_inputs() -> List:
    """
    Returns the initialization parameters for the Model constructor in the same order.
    """
    return [PAD, SCALE_FACTOR, POOL_KERNEL]