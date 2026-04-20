import torch
import torch.nn as nn
import torch

# Configuration
batch_size = 8
in_channels = 3
depth = 16
height = 64
width = 64
conv_out_channels = 64  # target channels for LazyConv3d
upsample_scale = 2

class Model(nn.Module):
    """
    Complex 3D -> 2D feature extractor that:
    - Applies a lazy-initialized 3D convolution over (D, H, W).
    - Activates with LeakyReLU.
    - Collapses the depth dimension via mean to produce 2D feature maps.
    - Upsamples the 2D maps bilinearly.
    - Computes a lightweight channel gating from the upsampled maps and applies it.
    - Final LeakyReLU for non-linearity.
    """
    def __init__(self,
                 out_channels: int = conv_out_channels,
                 upsample_scale: int = upsample_scale,
                 negative_slope: float = 0.02):
        super(Model, self).__init__()
        # LazyConv3d will infer in_channels on the first forward pass
        self.conv3d = nn.LazyConv3d(out_channels=out_channels, kernel_size=3, padding=1)
        self.prelu = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        # Bilinear upsampling for 2D feature maps (expects 4D input N,C,H,W)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=upsample_scale)
        # A small constant to stabilize gating multiplication
        self.eps = 1e-6
        # A second non-linearity after gating
        self.final_act = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        1. 3D convolution over (D,H,W)
        2. LeakyReLU activation
        3. Mean over depth -> 2D feature maps
        4. Bilinear upsampling (spatial)
        5. Channel gating computed from global spatial average
        6. Apply gating and final activation

        Args:
            x (torch.Tensor): Input tensor of shape (N, C_in, D, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (N, out_channels, H*scale, W*scale)
        """
        # 1. 3D conv
        z = self.conv3d(x)  # (N, out_channels, D, H, W)

        # 2. Non-linearity
        z = self.prelu(z)

        # 3. Collapse depth dimension by mean to produce 2D feature maps
        z2d = z.mean(dim=2)  # (N, out_channels, H, W)

        # 4. Bilinear upsampling to increase spatial resolution
        z_up = self.upsample(z2d)  # (N, out_channels, H*scale, W*scale)

        # 5. Compute a lightweight channel gating:
        #    - global spatial average per channel -> (N, C, 1, 1)
        #    - apply sigmoid to get gating factors in (0,1)
        channel_desc = z_up.mean(dim=(2, 3), keepdim=True)  # (N, out_channels, 1, 1)
        gate = torch.sigmoid(channel_desc)  # (N, out_channels, 1, 1)

        # 6. Apply gating (channel-wise) and a small residual-style scaling
        out = z_up * (gate + self.eps)

        # Final activation
        out = self.final_act(out)

        return out

def get_inputs():
    """
    Returns a list containing a single 5D tensor:
    shape = (batch_size, in_channels, depth, height, width)
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    No special initialization parameters required for LazyConv3d.
    The layer will infer in_channels from the first forward input.
    """
    return []