import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex 3D-to-2D processing module that:
    - Applies replication padding in 3D
    - Performs adaptive 3D average pooling
    - Aggregates depth dimension with mean and max branches
    - Concatenates the branches along channels and applies LazyInstanceNorm2d
    - Uses a 1x1 Conv2d to reduce channels and produces the final 2D feature map

    The module demonstrates combining nn.ReplicationPad3d, nn.AdaptiveAvgPool3d,
    and nn.LazyInstanceNorm2d into a single computation graph.
    """
    def __init__(
        self,
        in_channels: int,
        pad: tuple,
        adaptive_output: tuple,
        conv_out_channels: int,
        affine: bool = True,
        eps: float = 1e-5
    ):
        """
        Args:
            in_channels (int): Number of input channels for the 3D input.
            pad (tuple): ReplicationPad3d padding tuple of length 6:
                         (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back).
            adaptive_output (tuple): Output size for AdaptiveAvgPool3d as (D_out, H_out, W_out).
            conv_out_channels (int): Number of output channels for the final 1x1 Conv2d.
            affine (bool): If True, LazyInstanceNorm2d will have learnable affine parameters.
            eps (float): A value added to the denominator for numerical stability in InstanceNorm.
        """
        super(Model, self).__init__()
        # 3D replication padding
        self.pad3d = nn.ReplicationPad3d(pad)
        # Adaptive average pooling to a fixed (D_out, H_out, W_out)
        self.adaptive_pool3d = nn.AdaptiveAvgPool3d(adaptive_output)
        # Lazy instance norm for 2D feature maps: will be initialized on first forward
        self.inst_norm2d = nn.LazyInstanceNorm2d(affine=affine, eps=eps)
        # 1x1 conv to reduce channels after concatenation of mean/max branches
        # in_channels for conv2d is 2 * in_channels because of concat along channel dim
        self.reduce_conv2d = nn.Conv2d(in_channels=2 * in_channels, out_channels=conv_out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (N, conv_out_channels, H_out, W_out),
                          where H_out and W_out come from adaptive_output.
        """
        # Step 1: Replication padding in 3D
        x_padded = self.pad3d(x)

        # Step 2: Adaptive average pooling to fixed spatial output (D_out, H_out, W_out)
        pooled = self.adaptive_pool3d(x_padded)  # shape: (N, C, D_out, H_out, W_out)

        # Step 3: Aggregate depth dimension into two complementary branches
        # Branch A: Mean across depth
        depth_mean = pooled.mean(dim=2)  # shape: (N, C, H_out, W_out)
        # Branch B: Max across depth
        depth_max, _ = pooled.max(dim=2)  # shape: (N, C, H_out, W_out)

        # Step 4: Concatenate along channel dimension to combine features
        combined = torch.cat([depth_mean, depth_max], dim=1)  # shape: (N, 2*C, H_out, W_out)

        # Step 5: Normalize with LazyInstanceNorm2d (will set num_features lazily)
        normalized = self.inst_norm2d(combined)

        # Step 6: Non-linear activation
        activated = F.gelu(normalized)

        # Step 7: Channel reduction via 1x1 convolution and final sigmoid scaling
        reduced = self.reduce_conv2d(activated)
        output = torch.sigmoid(reduced)

        return output

# Configuration variables for creating inputs and initializing the model
batch_size = 8
channels = 16
depth = 12
height = 32
width = 32

# ReplicationPad3d expects (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
pad_sizes = (1, 1, 2, 2, 1, 1)
adaptive_output = (4, 8, 8)  # (D_out, H_out, W_out)
conv_out_channels = 16
affine = True
eps = 1e-5

def get_inputs():
    """
    Returns a list containing a single 5D input tensor suitable for the Model:
    shape (batch_size, channels, depth, height, width).
    """
    x = torch.randn(batch_size, channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters for Model in the correct order:
    in_channels, pad, adaptive_output, conv_out_channels, affine, eps
    """
    return [channels, pad_sizes, adaptive_output, conv_out_channels, affine, eps]