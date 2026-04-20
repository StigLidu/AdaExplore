import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class Model(nn.Module):
    """
    Complex 3D processing module combining InstanceNorm3d, FractionalMaxPool3d and Hardsigmoid.
    Pipeline:
      1. Instance normalization on input volumes.
      2. Fractional max pooling (returns pooled output and indices).
      3. Hardsigmoid activation on pooled features.
      4. Trilinear upsampling of activated pooled features back to input spatial size.
      5. Residual-style fusion with normalized input.
      6. Global spatial mean pooling and flatten to (N, C).

    The model demonstrates mixing normalization, fractional pooling, nonlinear activation,
    interpolation and reduction operations into a single forward pass.
    """
    def __init__(self, num_features: int, kernel_size: Tuple[int, int, int], output_ratio: Optional[Tuple[float, float, float]] = None):
        """
        Args:
            num_features (int): Number of channels in the input (C dimension).
            kernel_size (tuple): Kernel size for FractionalMaxPool3d (d, h, w).
            output_ratio (tuple, optional): Fractional output ratio for FractionalMaxPool3d.
                                             If None, pooling uses the kernel_size deterministically.
        """
        super(Model, self).__init__()
        self.num_features = num_features
        self.inst_norm = nn.InstanceNorm3d(num_features, affine=False, track_running_stats=False)
        # Use FractionalMaxPool3d with return_indices so we can inspect indices if needed
        # output_ratio is optional; if provided, controls fractional output size
        if output_ratio is not None:
            self.fmp = nn.FractionalMaxPool3d(kernel_size=kernel_size, output_ratio=output_ratio, return_indices=True)
        else:
            self.fmp = nn.FractionalMaxPool3d(kernel_size=kernel_size, return_indices=True)
        self.act = nn.Hardsigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (N, C) representing channel-wise global features.
        """
        # 1) Instance normalization (preserves shape)
        x_norm = self.inst_norm(x)

        # 2) Fractional max pooling: returns pooled tensor and indices
        pooled, indices = self.fmp(x_norm)

        # 3) Non-linear activation on pooled output
        pooled_act = self.act(pooled)

        # 4) Upsample pooled activated features back to original spatial resolution
        # Use trilinear interpolation; note align_corners=False is safe for most sizes
        upsampled = F.interpolate(pooled_act, size=x.shape[2:], mode='trilinear', align_corners=False)

        # 5) Fuse normalized input and upsampled pooled features (residual-like addition)
        fused = x_norm + upsampled

        # 6) Global spatial mean pooling over D, H, W -> shape (N, C, 1, 1, 1)
        global_mean = fused.mean(dim=(2, 3, 4), keepdim=True)

        # 7) Flatten spatial dimensions to produce (N, C)
        out = global_mean.view(global_mean.shape[0], global_mean.shape[1])

        return out

# Configuration (module-level)
batch_size = 8
channels = 32
depth = 20
height = 48
width = 40
kernel_size = (2, 3, 2)                 # kernel sizes for fractional pooling
output_ratio = (0.6, 0.5, 0.7)          # fractional output ratio for pooling

def get_inputs():
    """
    Returns:
        List containing a single 5D input tensor shaped (N, C, D, H, W)
    """
    x = torch.randn(batch_size, channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization inputs required to construct the Model:
      - num_features (int)
      - kernel_size (tuple)
      - output_ratio (tuple)
    """
    return [channels, kernel_size, output_ratio]