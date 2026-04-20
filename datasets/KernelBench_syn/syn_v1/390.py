import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex module that combines channel-wise softmax gating, upsampling, and hardtanh clipping.
    
    Computation pattern:
      1. Project input channels to a higher-dimensional embedding with a 1x1 convolution.
      2. Compute a channel-wise softmax at each spatial location to act as an attention/gating map.
      3. Apply the gating to the projected features (element-wise multiplication).
      4. Upsample the gated features spatially.
      5. Compute a spatial summary (mean over channels), concatenate it as an extra channel.
      6. Project back to the original number of channels with a 1x1 convolution.
      7. Apply HardTanh activation to bound the outputs.
    """
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        up_scale: int = 2,
        hardtanh_min: float = -1.0,
        hardtanh_max: float = 1.0
    ):
        """
        Initializes the model components.

        Args:
            in_channels (int): Number of channels in the input tensor.
            mid_channels (int): Number of intermediate channels used for projection/gating.
            up_scale (int): Integer upsampling scale factor for spatial dimensions.
            hardtanh_min (float): Minimum value for Hardtanh activation.
            hardtanh_max (float): Maximum value for Hardtanh activation.
        """
        super(Model, self).__init__()
        # 1x1 conv to project input channels -> mid_channels
        self.proj_in = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=True)
        # softmax over channel dimension for each spatial location
        self.channel_softmax = nn.Softmax2d()
        # Upsample spatially by integer factor
        self.upsample = nn.Upsample(scale_factor=up_scale, mode='bilinear', align_corners=False)
        # After concat we will have mid_channels + 1 channels (pooled summary)
        self.proj_out = nn.Conv2d(mid_channels + 1, in_channels, kernel_size=1, bias=True)
        # Hardtanh to clip outputs into a bounded range
        self.hardtanh = nn.Hardtanh(min_val=hardtanh_min, max_val=hardtanh_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining projection, channel gating, upsampling, and clipping.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (N, C_in, H * up_scale, W * up_scale), clipped.
        """
        # Project channels to mid_channels (1x1 conv)
        proj = self.proj_in(x)                        # shape: (N, mid_channels, H, W)

        # Compute channel-wise softmax per spatial location (attention/gating)
        gate = self.channel_softmax(proj)             # shape: (N, mid_channels, H, W)

        # Apply gating
        gated = proj * gate                           # element-wise re-weighting

        # Upsample gated features spatially
        up = self.upsample(gated)                     # shape: (N, mid_channels, H*scale, W*scale)

        # Spatial summary: mean across channels -> one extra channel
        spatial_summary = up.mean(dim=1, keepdim=True)  # shape: (N, 1, H*scale, W*scale)

        # Concatenate the summary as an additional channel
        merged = torch.cat([up, spatial_summary], dim=1)  # shape: (N, mid_channels+1, H*, W*)

        # Project back to original input channels
        out = self.proj_out(merged)                   # shape: (N, in_channels, H*, W*)

        # Apply HardTanh clipping and return
        return self.hardtanh(out)

# Configuration / default sizes
batch_size = 8
in_channels = 32
mid_channels = 64
height = 64
width = 64
up_scale = 2
hardtanh_min = -3.0
hardtanh_max = 3.0

def get_inputs():
    """
    Returns a list with a single input tensor matching the configured shapes:
      - Tensor shape: (batch_size, in_channels, height, width)
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for the Model constructor:
      [in_channels, mid_channels, up_scale, hardtanh_min, hardtanh_max]
    """
    return [in_channels, mid_channels, up_scale, hardtanh_min, hardtanh_max]