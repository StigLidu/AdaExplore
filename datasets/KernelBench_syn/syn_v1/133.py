import torch
import torch.nn as nn
from typing import List, Any

class Model(nn.Module):
    """
    Complex 3D feature extractor that combines Conv3d, Instance Normalization (or Lazy variant),
    Softsign non-linearity, and global spatial pooling followed by a fully-connected projection.

    Computation pattern (forward):
      1. 3D convolution to produce hidden feature maps.
      2. Instance normalization (or lazy instance norm) on feature maps.
      3. Softsign activation applied element-wise.
      4. Compute spatial max and spatial mean over (D,H,W).
      5. Concatenate pooled statistics (max, mean) along channel axis.
      6. Linear projection to desired output dimension.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_features: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        affine: bool = True,
        use_lazy_instancenorm: bool = False
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            hidden_channels (int): Number of output channels from the convolution and InstanceNorm.
            out_features (int): Number of output features from the final linear layer.
            kernel_size (int, optional): Kernel size for Conv3d. Defaults to 3.
            stride (int, optional): Stride for Conv3d. Defaults to 1.
            padding (int, optional): Padding for Conv3d. Defaults to 1.
            affine (bool, optional): Whether InstanceNorm3d has learnable affine parameters. Defaults to True.
            use_lazy_instancenorm (bool, optional): Use LazyInstanceNorm3d (num_features inferred at first forward).
        """
        super(Model, self).__init__()
        # Convolution to produce hidden feature maps
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=hidden_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

        # Choose between eager InstanceNorm3d (requires known hidden_channels) or LazyInstanceNorm3d
        if use_lazy_instancenorm:
            # LazyInstanceNorm3d defers num_features until first forward
            self.instnorm = nn.LazyInstanceNorm3d(affine=affine)
        else:
            self.instnorm = nn.InstanceNorm3d(num_features=hidden_channels, affine=affine)

        # Softsign activation (element-wise)
        self.softsign = nn.Softsign()

        # Final fully connected projection: input dim is 2 * hidden_channels (max + mean)
        self.fc = nn.Linear(hidden_channels * 2, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C_in, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (N, out_features) after pooling and projection.
        """
        # 1) Convolution -> (N, hidden, D', H', W')
        x_conv = self.conv(x)

        # 2) Instance normalization (or lazy instance norm)
        x_norm = self.instnorm(x_conv)

        # 3) Softsign activation
        x_act = self.softsign(x_norm)

        # 4) Global spatial statistics: max and mean over D, H, W
        # Use torch.amax for numerical clarity (equivalent to x_act.max over dims)
        spatial_max = torch.amax(x_act, dim=(2, 3, 4))   # shape: (N, hidden)
        spatial_mean = torch.mean(x_act, dim=(2, 3, 4))  # shape: (N, hidden)

        # 5) Concatenate statistics and project
        combined = torch.cat([spatial_max, spatial_mean], dim=1)  # shape: (N, 2*hidden)
        out = self.fc(combined)  # shape: (N, out_features)

        return out

# Module-level configuration variables (default example values)
batch_size = 8
in_channels = 3
hidden_channels = 32
depth = 16
height = 32
width = 32
kernel_size = 3
stride = 1
padding = 1
out_features = 64
affine = True
use_lazy_instancenorm = False

def get_inputs() -> List[torch.Tensor]:
    """
    Returns a list of input tensors suitable for the Model forward pass.
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs() -> List[Any]:
    """
    Returns the initialization parameters for the Model constructor in the same order:
      [in_channels, hidden_channels, out_features, kernel_size, stride, padding, affine, use_lazy_instancenorm]
    """
    return [in_channels, hidden_channels, out_features, kernel_size, stride, padding, affine, use_lazy_instancenorm]