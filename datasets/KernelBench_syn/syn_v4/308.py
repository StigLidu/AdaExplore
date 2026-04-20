import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex model that mixes 3D instance normalization, pixel unshuffle on spatial slices,
    a pointwise convolution, lazy 1D instance normalization across a depth-like dimension,
    and a final linear projection.

    Pipeline:
    1. InstanceNorm3d over (N, C, D, H, W)
    2. Merge batch and depth to form (N*D, C, H, W)
    3. PixelUnshuffle to reduce spatial resolution and increase channels
    4. 1x1 Conv2d to mix the increased channels -> conv_out channels
    5. Global spatial average pooling -> (N*D, conv_out)
    6. Reshape to (N, conv_out, D) and apply LazyInstanceNorm1d (learns num_features lazily)
    7. ReLU, mean across depth, then final Linear projection
    """
    def __init__(self,
                 in_channels: int,
                 downscale_factor: int = 2,
                 conv_out_channels: int = 32,
                 linear_out_features: int = 10):
        """
        Args:
            in_channels (int): Number of input channels for the 3D tensor.
            downscale_factor (int): PixelUnshuffle downscale factor (must divide H and W).
            conv_out_channels (int): Output channels of the 1x1 Conv2d after PixelUnshuffle.
            linear_out_features (int): Final output feature dimension after linear layer.
        """
        super(Model, self).__init__()
        # Normalize across channels for each instance in 3D input
        self.inst3d = nn.InstanceNorm3d(num_features=in_channels, affine=False, track_running_stats=False)
        # PixelUnshuffle reduces H,W by factor and increases channels by factor^2
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor)
        # 1x1 conv to mix the expanded channels after PixelUnshuffle
        self.conv1x1 = nn.Conv2d(in_channels=in_channels * (downscale_factor ** 2),
                                 out_channels=conv_out_channels,
                                 kernel_size=1,
                                 bias=True)
        # Lazy instance norm in 1D will infer the num_features (channels) on first forward
        self.lazy_norm1d = nn.LazyInstanceNorm1d()
        # Final projection
        self.linear = nn.Linear(conv_out_channels, linear_out_features)
        # Store some parameters for use in forward/reshape logic
        self.in_channels = in_channels
        self.downscale_factor = downscale_factor
        self.conv_out_channels = conv_out_channels
        self.linear_out_features = linear_out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (N, linear_out_features).
        """
        # x: (N, C, D, H, W)
        N, C, D, H, W = x.shape
        # 1) 3D instance normalization
        y = self.inst3d(x)  # (N, C, D, H, W)

        # 2) Merge depth into batch dimension to apply PixelUnshuffle slice-wise
        # permute to (N, D, C, H, W) then view to (N*D, C, H, W)
        y = y.permute(0, 2, 1, 3, 4).contiguous()
        y = y.view(N * D, C, H, W)  # (N*D, C, H, W)

        # 3) PixelUnshuffle: (N*D, C, H, W) -> (N*D, C * r^2, H/r, W/r)
        y = self.pixel_unshuffle(y)

        # 4) 1x1 conv to mix channels: (N*D, conv_out_channels, H2, W2)
        y = self.conv1x1(y)

        # 5) Global spatial average pool -> (N*D, conv_out_channels)
        y = y.mean(dim=[2, 3])  # avg over H2 and W2

        # 6) Reshape back to (N, conv_out_channels, D) to apply LazyInstanceNorm1d across depth
        y = y.view(N, D, self.conv_out_channels).permute(0, 2, 1).contiguous()  # (N, conv_out_channels, D)

        # LazyInstanceNorm1d will infer num_features (= conv_out_channels) on first call
        y = self.lazy_norm1d(y)  # (N, conv_out_channels, D)

        # 7) Activation and aggregation over depth -> (N, conv_out_channels)
        y = torch.relu(y)
        y = y.mean(dim=2)

        # 8) Final linear projection -> (N, linear_out_features)
        out = self.linear(y)
        return out

# Configuration / initialization defaults
batch_size = 8
in_channels = 16
depth = 5
height = 64
width = 64
downscale_factor = 2  # must divide height and width
conv_out_channels = 32
linear_out_features = 10

def get_inputs():
    """
    Returns example input tensors for the model.

    Output:
        [x] where x is a tensor of shape (batch_size, in_channels, depth, height, width)
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters for the Model constructor in order:
      in_channels, downscale_factor, conv_out_channels, linear_out_features
    """
    return [in_channels, downscale_factor, conv_out_channels, linear_out_features]