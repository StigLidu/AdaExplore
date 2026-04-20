import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex image-processing style module that:
    - Applies PixelUnshuffle to reduce spatial resolution and increase channels.
    - Normalizes each instance with InstanceNorm2d.
    - Applies a non-linearity followed by LocalResponseNorm across channels.
    - Pools spatially to a vector and projects to an output embedding via a Linear layer.

    This creates a different computation pattern from simple activations or reductions by combining
    pixel reorganization with normalization across both spatial-instance and local-channel contexts.
    """
    def __init__(self, in_channels: int, unshuffle_factor: int, out_features: int, lrn_size: int = 5):
        """
        Args:
            in_channels (int): Number of input channels.
            unshuffle_factor (int): Downscale factor for PixelUnshuffle. Must evenly divide spatial dims.
            out_features (int): Dimensionality of the final output vector per example.
            lrn_size (int): LocalResponseNorm window size (number of channels to sum over).
        """
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.unshuffle_factor = unshuffle_factor
        self.out_features = out_features
        self.lrn_size = lrn_size

        # PixelUnshuffle will increase channels by unshuffle_factor^2
        self.channels_post = in_channels * (unshuffle_factor ** 2)
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=unshuffle_factor)

        # InstanceNorm2d normalizes per-sample per-channel statistics (C must equal channels_post)
        # Use affine=True to allow learned scale/shift after normalization.
        self.inst_norm = nn.InstanceNorm2d(self.channels_post, affine=True, track_running_stats=False)

        # LocalResponseNorm operates across channels for each spatial location.
        # Common hyperparameters for LRN: alpha=1e-4, beta=0.75, k=1.0
        self.lrn = nn.LocalResponseNorm(size=lrn_size, alpha=1e-4, beta=0.75, k=1.0)

        # Spatial pooling to 1x1 so we can linearly project the per-channel descriptors
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Final linear projection from channels_post to out_features
        self.fc = nn.Linear(self.channels_post, self.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_channels, H, W) where H and W
                              are divisible by unshuffle_factor.

        Returns:
            torch.Tensor: Output tensor of shape (batch, out_features).
        """
        # 1) Reorganize pixels: (B, C, H, W) -> (B, C*r^2, H/r, W/r)
        x = self.pixel_unshuffle(x)

        # 2) Instance normalization across each sample
        x = self.inst_norm(x)

        # 3) Non-linearity
        x = F.relu(x, inplace=True)

        # 4) Local response normalization across channels (per spatial location)
        x = self.lrn(x)

        # 5) Spatial global descriptor via adaptive average pooling -> (B, C', 1, 1)
        x = self.avgpool(x)

        # 6) Flatten and project to output embedding
        x = x.view(x.size(0), -1)  # (B, channels_post)
        out = self.fc(x)           # (B, out_features)
        return out

# Module-level configuration
batch_size = 8
in_channels = 3
height = 64
width = 64
unshuffle_factor = 2  # Must divide height and width
out_features = 128
lrn_size = 5

def get_inputs():
    """
    Creates a random input tensor matching the configured batch and spatial sizes.
    Returns the list of inputs consumed by Model.forward.
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns the initialization arguments for the Model constructor.
    """
    return [in_channels, unshuffle_factor, out_features, lrn_size]