import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

"""
Complex PyTorch kernel module that:
- Applies a spatial Softmax2d across channel dimension for each depth slice.
- Produces weighted features by elementwise multiplication.
- Performs a 3D max-pooling (with indices) and a corresponding MaxUnpool3d.
- Uses an AvgPool3d layer (kernel_size=1) to demonstrate incorporation of the layer API.
- Adds a residual connection and reduces to per-channel descriptors via global mean.

Module-level configuration variables control tensor shapes to ensure compatibility.
"""

# Configuration (ensure divisibility for pooling)
batch_size = 4
channels = 16
depth = 8    # should be >= 2 and divisible by 2 for pooling/unpooling
height = 32  # should be >= 2 and divisible by 2
width = 32   # should be >= 2 and divisible by 2

class Model(nn.Module):
    """
    Model combining Softmax2d, AvgPool3d, and MaxUnpool3d with a max_pool3d to obtain indices.
    The forward pass:
      1. Compute spatial softmax across channels for each depth slice.
      2. Weight original features by that softmax.
      3. MaxPool3d (with indices) to downsample.
      4. AvgPool3d (identity smoothing via kernel_size=1 here) to illustrate usage.
      5. MaxUnpool3d to restore original spatial dimensions.
      6. Residual addition with the input and global mean pooling to produce (N, C) output.
    """
    def __init__(self):
        super(Model, self).__init__()
        # Softmax2d operates on 4D tensors (N, C, H, W). We'll reshape our 5D input accordingly.
        self.softmax2d = nn.Softmax2d()
        # Average pooling layer (here kernel_size=1 to preserve pooled shape; used to include the API)
        self.avgpool3d = nn.AvgPool3d(kernel_size=1)
        # MaxUnpool3d to invert max pooling (must match the kernel/stride used in forward's max_pool3d)
        self.unpool3d = nn.MaxUnpool3d(kernel_size=2, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, D, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (N, C) containing per-channel descriptors.
        """
        # Expect shape (N, C, D, H, W)
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input (N,C,D,H,W), got shape {x.shape}")

        N, C, D, H, W = x.shape

        # 1) Compute Softmax2d across channel dimension for each depth slice.
        #    Rearrange to (N*D, C, H, W) so Softmax2d applies per-spatial-location across channels.
        x_perm = x.permute(0, 2, 1, 3, 4)           # (N, D, C, H, W)
        x_resh = x_perm.reshape(N * D, C, H, W)     # (N*D, C, H, W)
        weights_resh = self.softmax2d(x_resh)       # (N*D, C, H, W)
        # Restore to (N, C, D, H, W)
        weights = weights_resh.reshape(N, D, C, H, W).permute(0, 2, 1, 3, 4)

        # 2) Weighted features
        weighted = x * weights                       # (N, C, D, H, W)

        # 3) MaxPool3d with indices to downsample (will produce indices for unpool)
        #    Using kernel_size=2 and stride=2 to downsample spatially and in depth.
        pooled, indices = F.max_pool3d(weighted, kernel_size=2, stride=2, padding=0, return_indices=True)

        # 4) AvgPool3d - here kernel_size=1 to preserve shape (demonstrates usage of the layer)
        smoothed = self.avgpool3d(pooled)

        # 5) MaxUnpool3d to restore the original size. Provide output_size to be safe.
        unpooled = self.unpool3d(smoothed, indices, output_size=x.size())

        # 6) Residual connection and global mean pooling across D,H,W -> produce (N, C)
        out = unpooled + x
        out_mean = out.mean(dim=[2, 3, 4])  # mean over depth, height, width

        return out_mean

def get_inputs() -> List[torch.Tensor]:
    """
    Returns:
        A list with a single input tensor of shape (batch_size, channels, depth, height, width)
    """
    x = torch.randn(batch_size, channels, depth, height, width)
    return [x]

def get_init_inputs() -> List:
    """
    Return any initialization inputs required by the module.
    This model uses fixed layer configurations, so no init parameters are required.
    """
    return []