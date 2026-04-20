import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex 3D upsampling block using ConvTranspose3d combined with channel gating and clamping.
    
    Computation steps:
    1. Apply a transposed 3D convolution to upsample the input spatially.
    2. Apply Hardswish activation element-wise.
    3. Compute a channel-wise descriptor by spatial global average, clamp it with Hardtanh,
       transform to a gating vector with sigmoid, and apply it to the activated features.
    4. Upsample the original input (trilinear) to match the transposed conv output and add it (residual).
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 output_padding: int = 0):
        super(Model, self).__init__()
        # Transposed convolution to increase spatial resolution
        self.deconv = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=True
        )
        # Non-linearities
        self.hardswish = nn.Hardswish()
        # Hardtanh to clamp the channel descriptor before sigmoid gating
        self.hardtanh = nn.Hardtanh(min_val=-0.5, max_val=0.5)
        # Small learnable bias applied after gating to allow subtle shifts
        self.post_bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_channels, D, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (batch, out_channels, D*stride, H*stride, W*stride)
        """
        # 1) Upsample with transposed convolution
        y = self.deconv(x)  # shape: (B, out_channels, D_out, H_out, W_out)

        # 2) Non-linearity
        y_act = self.hardswish(y)

        # 3) Channel-wise gating:
        #    - Global average pooling across spatial dims -> (B, C, 1, 1, 1)
        #    - Clamp descriptor with Hardtanh -> limit extreme values
        #    - Sigmoid to produce gating in (0,1)
        ch_desc = y_act.mean(dim=(2, 3, 4), keepdim=True)            # (B, C, 1, 1, 1)
        ch_clamped = self.hardtanh(ch_desc)                           # (B, C, 1, 1, 1)
        gate = torch.sigmoid(ch_clamped)                              # (B, C, 1, 1, 1)
        y_gated = y_act * gate                                        # (B, C, D_out, H_out, W_out)

        # 4) Residual connection: upsample original input to match spatial dims and add
        #    Use trilinear interpolation for a smooth residual
        target_spatial = y_gated.shape[2:]
        x_upsampled = F.interpolate(x, size=target_spatial, mode='trilinear', align_corners=False)
        # If channel dimensions differ, project residual channels by a simple mean-based mapping:
        # compute per-channel contribution by averaging input channels and then expand to out_channels
        if x_upsampled.shape[1] != y_gated.shape[1]:
            # Global average across channels to get a single-channel residual map, then expand
            residual_map = x_upsampled.mean(dim=1, keepdim=True)  # (B,1,D_out,H_out,W_out)
            residual = residual_map.expand(-1, y_gated.shape[1], -1, -1, -1)  # broadcast to out_channels
        else:
            residual = x_upsampled

        out = y_gated + residual + self.post_bias  # simple learnable shift per channel

        return out

# Configuration / default sizes
batch_size = 2
in_channels = 8
out_channels = 16
D = 8
H = 8
W = 8

# ConvTranspose3d parameters chosen to double spatial dimensions
kernel_size = 4
stride = 2
padding = 1
output_padding = 0

def get_inputs():
    """
    Returns the input tensors for the model:
    - x: random input tensor of shape (batch_size, in_channels, D, H, W)
    """
    x = torch.randn(batch_size, in_channels, D, H, W)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters for the Model constructor in the following order:
    [in_channels, out_channels, kernel_size, stride, padding, output_padding]
    """
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding]