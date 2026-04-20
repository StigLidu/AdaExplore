import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex model combining UpsamplingNearest2d, SyncBatchNorm, and InstanceNorm3d.
    
    Pipeline:
      1. Upsample the input using nearest neighbor interpolation.
      2. Apply SyncBatchNorm across channels.
      3. Apply a 3x3 convolution + ReLU.
      4. Reshape the 4D tensor into a 5D tensor by splitting channels into (channels//depth, depth).
      5. Apply InstanceNorm3d to normalize over each pseudo-volume.
      6. Merge the depth dimension back into channels and fuse with the upsampled tensor via elementwise multiplication.
      7. Final pointwise convolution to produce the output.
    """
    def __init__(self, in_channels: int, depth: int = 4, scale_factor: int = 2):
        """
        Args:
            in_channels (int): Number of channels in the input tensor (must be divisible by depth).
            depth (int): Number of slices to create in the pseudo-depth dimension.
            scale_factor (int): Upsampling scale factor for spatial dimensions.
        """
        super(Model, self).__init__()
        if in_channels % depth != 0:
            raise ValueError("in_channels must be divisible by depth")
        self.in_channels = in_channels
        self.depth = depth
        self.scale_factor = scale_factor

        # Layers from the provided list plus standard conv/activation layers
        self.upsample = nn.UpsamplingNearest2d(scale_factor=scale_factor)
        self.sync_bn = nn.SyncBatchNorm(num_features=in_channels)
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        # After splitting channels into (C3, D), C3 = in_channels // depth
        c3 = in_channels // depth
        self.inst_norm3d = nn.InstanceNorm3d(num_features=c3, affine=False, eps=1e-5)
        
        # Final 1x1 conv to mix channels again
        self.pointwise = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W) where C == in_channels.

        Returns:
            torch.Tensor: Output tensor of shape (N, C, H*scale, W*scale).
        """
        # 1) Upsample
        x_up = self.upsample(x)  # (N, C, H*s, W*s)

        # 2) SyncBatchNorm
        x_bn = self.sync_bn(x_up)

        # 3) Conv + ReLU
        x_conv = self.conv3x3(x_bn)
        x_act = self.relu(x_conv)

        # 4) Reshape to 5D: (N, C3, D, H', W')
        N, C, Hs, Ws = x_act.shape
        D = self.depth
        C3 = C // D
        x_5d = x_act.view(N, C3, D, Hs, Ws)

        # 5) InstanceNorm3d
        x_norm3d = self.inst_norm3d(x_5d)

        # 6) Merge depth back into channels: (N, C3*D, H', W')
        x_merged = x_norm3d.view(N, C3 * D, Hs, Ws)

        # 7) Fuse with the upsampled tensor and final pointwise conv
        # Elementwise multiply to create an interaction between features
        fused = x_merged * x_up
        out = self.pointwise(fused)

        return out

# Configuration variables
batch_size = 8
in_channels = 32  # Must be divisible by depth
height = 64
width = 64
depth = 4
scale_factor = 2

def get_inputs():
    """
    Returns a list with a single input tensor matching the configuration.
    Shape: (batch_size, in_channels, height, width)
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for the Model: [in_channels, depth, scale_factor]
    """
    return [in_channels, depth, scale_factor]