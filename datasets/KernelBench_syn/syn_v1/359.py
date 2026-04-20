import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex module combining a lazy 3D convolution, 3D average pooling,
    dimensionality collapse, and 2D nearest-neighbor upsampling to fuse
    two modalities (3D volume and 2D feature map). A simple channel-wise
    gating is applied at the end.
    """
    def __init__(self, conv_out_channels: int = 16, up_scale: int = 2):
        """
        Args:
            conv_out_channels (int): Number of output channels for the LazyConv3d.
            up_scale (int): Nearest-neighbor upsampling scale factor for the 2D feature map.
        """
        super(Model, self).__init__()
        # LazyConv3d will infer in_channels at the first forward pass
        self.conv3d = nn.LazyConv3d(out_channels=conv_out_channels, kernel_size=(3, 3, 3), padding=1)
        # 3D average pooling reduces spatial and depth resolution by half
        self.pool3d = nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # 2D nearest neighbor upsampling to spatially match the pooled/collapsed 3D tensor
        self.upsample2d = nn.UpsamplingNearest2d(scale_factor=up_scale)
        # small learnable scale for gating to increase modeling capacity (initialized near 1)
        self.gate_scale = nn.Parameter(torch.tensor(1.0))
        self.up_scale = up_scale
        self.conv_out_channels = conv_out_channels

    def forward(self, x3d: torch.Tensor, x2d: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
         - x3d: (N, C3, D, H, W)
         - x2d: (N, C2, h, w)  where (h * up_scale, w * up_scale) == (H//2, W//2)

        Processing steps:
         1. Apply lazy 3D convolution to x3d -> (N, conv_out_channels, D, H, W)
         2. Apply 3D average pooling to downsample depth and spatial dims by 2 -> (N, conv_out_channels, D//2, H//2, W//2)
         3. Collapse depth via mean -> (N, conv_out_channels, H//2, W//2)
         4. Upsample x2d to match spatial dims -> (N, C2, H//2, W//2)
         5. Concatenate along channel dim -> (N, conv_out_channels + C2, H//2, W//2)
         6. Compute per-channel global context (spatial mean), apply a sigmoid gating scaled by a learnable parameter,
            and gate the concatenated feature map.
        """
        # 1. 3D conv
        y = self.conv3d(x3d)  # (N, conv_out_channels, D, H, W)

        # 2. 3D average pooling (reduces D, H, W by factor of 2)
        y = self.pool3d(y)  # (N, conv_out_channels, D//2, H//2, W//2)

        # 3. Collapse depth dimension by averaging over depth
        y_collapsed = torch.mean(y, dim=2)  # (N, conv_out_channels, H//2, W//2)

        # 4. Upsample x2d to match the spatial dims of y_collapsed
        x2d_up = self.upsample2d(x2d)  # (N, C2, H//2, W//2)

        # If spatial sizes don't match due to rounding, center-crop/pad to match exactly
        _, _, Hc, Wc = y_collapsed.shape
        _, _, Hu, Wu = x2d_up.shape
        if Hu != Hc or Wu != Wc:
            # crop or pad x2d_up centrally to match (simple handling)
            # Calculate cropping indices
            start_h = max(0, (Hu - Hc) // 2)
            start_w = max(0, (Wu - Wc) // 2)
            end_h = start_h + Hc
            end_w = start_w + Wc
            x2d_up = x2d_up[:, :, start_h:end_h, start_w:end_w]
            # If smaller, pad
            if x2d_up.shape[2] != Hc or x2d_up.shape[3] != Wc:
                pad_h = Hc - x2d_up.shape[2]
                pad_w = Wc - x2d_up.shape[3]
                pad = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
                x2d_up = F.pad(x2d_up, pad)

        # 5. Concatenate along channel dim
        z = torch.cat([y_collapsed, x2d_up], dim=1)  # (N, conv_out_channels + C2, H//2, W//2)

        # 6. Channel-wise gating: compute spatial global mean per channel, apply sigmoid and scale
        channel_context = torch.mean(z, dim=(2, 3), keepdim=True)  # (N, C_total, 1, 1)
        gate = torch.sigmoid(self.gate_scale * channel_context)    # (N, C_total, 1, 1)
        out = z * gate  # gated feature map, same shape as z

        return out


# Configuration / default sizes
batch_size = 8
in_channels_3d = 3   # will be inferred by LazyConv3d at first forward
depth = 8
height = 32
width = 24

in_channels_2d = 12
small_h = 8   # after upsampling by up_scale=2 -> 16 == height//2
small_w = 6   # after upsampling by up_scale=2 -> 12 == width//2

conv_out_channels = 16
up_scale = 2

def get_inputs():
    """
    Returns a list of input tensors [x3d, x2d] matching the model's expected shapes.
    x3d: (batch_size, in_channels_3d, depth, height, width)
    x2d: (batch_size, in_channels_2d, small_h, small_w)
    """
    x3d = torch.randn(batch_size, in_channels_3d, depth, height, width)
    x2d = torch.randn(batch_size, in_channels_2d, small_h, small_w)
    return [x3d, x2d]

def get_init_inputs():
    """
    Returns the initialization parameters for the Model constructor.
    """
    return [conv_out_channels, up_scale]