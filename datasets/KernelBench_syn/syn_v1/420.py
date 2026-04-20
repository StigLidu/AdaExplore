import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex 2D feature transform that combines PReLU activation, 1x1 channel expansion,
    spatial MaxPool branch, RReLU branch, upsampling, concatenation, and a final conv
    with residual connection.

    Computation steps (high-level):
      1. Apply channel-wise PReLU to input.
      2. Expand channels with a 1x1 convolution.
      3. Split expanded tensor into two channel-halves.
      4. Branch A: MaxPool2d spatially reduces features, then upsample back.
      5. Branch B: Apply randomized leaky activation (RReLU).
      6. Concatenate branches along channel dim and fuse with a 3x3 convolution.
      7. Add residual connection and return output.
    """
    def __init__(self, in_channels: int, pool_kernel: int = 2, pool_stride: int = 2):
        """
        Args:
            in_channels (int): Number of input channels (must be even for channel split).
            pool_kernel (int): Kernel size for MaxPool2d.
            pool_stride (int): Stride for MaxPool2d.
        """
        super(Model, self).__init__()
        if in_channels % 2 != 0:
            raise ValueError("in_channels must be even for an even channel split.")
        self.in_channels = in_channels
        self.expanded_channels = in_channels * 2
        # Channel-wise learnable negative slopes
        self.prelu = nn.PReLU(num_parameters=in_channels)
        # Randomized leaky ReLU for second branch
        self.rrelu = nn.RReLU(lower=0.125, upper=0.333, inplace=False)
        # 1x1 conv to increase capacity and enable channel-wise mixing
        self.conv1x1 = nn.Conv2d(in_channels, self.expanded_channels, kernel_size=1, bias=True)
        # MaxPool for branch A
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride, padding=0, ceil_mode=False)
        # Fusion conv to mix concatenated features
        self.fusion_conv = nn.Conv2d(self.expanded_channels, in_channels, kernel_size=3, padding=1, bias=True)
        # A small channel re-scaling after global context
        self.gate_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),          # B x C' x 1 x 1
            nn.Conv2d(in_channels, in_channels, kernel_size=1),  # B x C x 1 x 1
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W), where C == in_channels.

        Returns:
            torch.Tensor: Output tensor of shape (B, in_channels, H, W).
        """
        # Save residual
        residual = x

        # 1) Channel-wise parametric activation
        x_act = self.prelu(x)  # (B, C, H, W)

        # 2) Expand channels (1x1 conv)
        x_exp = self.conv1x1(x_act)  # (B, 2C, H, W)

        # 3) Split into two halves along channel dimension
        C2 = x_exp.shape[1]
        half = C2 // 2
        left = x_exp[:, :half, :, :]   # Branch A
        right = x_exp[:, half:, :, :]  # Branch B

        # 4) Branch A: spatial reduction and upsample back to original spatial size
        pooled = self.pool(left)  # (B, C, Hp, Wp)
        # Upsample to match right's spatial dimensions (which are original H, W)
        upsampled = F.interpolate(pooled, size=right.shape[-2:], mode="bilinear", align_corners=False)  # (B, C, H, W)

        # 5) Branch B: randomized leaky activation
        right_activated = self.rrelu(right)  # (B, C, H, W)

        # 6) Concatenate along channel dim and fuse
        fused = torch.cat([upsampled, right_activated], dim=1)  # (B, 2C, H, W)
        fused = self.fusion_conv(fused)  # (B, C, H, W)

        # 7) Global gating (channel-wise re-weight) and residual addition
        gate = self.gate_conv(fused)  # (B, C, 1, 1)
        out = fused * gate + residual  # broadcast (B, C, H, W)

        return out

# Configuration / default sizes
batch_size = 8
in_channels = 64  # Must be even
height = 128
width = 128
pool_kernel = 2
pool_stride = 2

def get_inputs():
    """
    Returns:
        list: [input_tensor] where input_tensor has shape (batch_size, in_channels, height, width)
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Initialization parameters for Model constructor.

    Returns:
        list: [in_channels, pool_kernel, pool_stride]
    """
    return [in_channels, pool_kernel, pool_stride]