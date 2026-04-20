import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# Configuration / default values
batch_size = 8
in_channels = 12
height = 64   # keep even to ensure pooling/upsample match
width = 64    # keep even to ensure pooling/upsample match
eps_default = 1e-5

class Model(nn.Module):
    """
    A moderately complex module demonstrating a mix of padding layers, a depthwise smoothing
    convolution, a channel-mixing pointwise convolution, SELU activation, and RMS-style normalization.
    The model uses CircularPad2d to implement wrap-around convolution behavior and ConstantPad2d
    to demonstrate constant-valued margins which are then cropped back to the original size.
    """
    def __init__(self, in_channels: int, eps: float = eps_default):
        """
        Args:
            in_channels (int): Number of channels in the input tensor.
            eps (float): Small epsilon used for RMS normalization.
        """
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.eps = eps

        # Circular padding will be applied before a 3x3 depthwise conv so the conv sees wrap-around neighbors.
        self.circ_pad = nn.CircularPad2d(1)  # pad 1 on all sides (left, right, top, bottom)

        # Depthwise 3x3 convolution - used as a fixed local smoother (initialized as average filter)
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, groups=in_channels, bias=False)

        # Initialize the depthwise conv as an average filter and freeze it (non-learnable smoothing)
        with torch.no_grad():
            # weight shape: (in_channels, 1, 3, 3)
            avg_kernel = torch.full((1, 3, 3), 1.0 / 9.0, dtype=torch.float32)
            self.depthwise.weight.data = avg_kernel.repeat(in_channels, 1, 1, 1)
            self.depthwise.weight.requires_grad = False

        # SELU non-linearity instance
        self.selu = nn.SELU()

        # A 1x1 pointwise convolution to mix channels after downsampling
        self.pointwise = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)

        # Constant pad to demonstrate constant margin addition; using zero value by default
        self.const_pad = nn.ConstantPad2d(1, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        1. Apply circular padding so the 3x3 depthwise conv effectively wraps around edges.
        2. Depthwise smoothing conv -> SELU activation.
        3. Downsample (average pooling) to capture coarser features.
        4. Pointwise conv mixes channels at the lower resolution.
        5. Upsample back to original spatial resolution.
        6. Apply constant padding and crop center to demonstrate ConstantPad2d usage.
        7. Element-wise combine with the original input (residual-like multiplicative merge).
        8. Apply RMS normalization across channels (keeping dims).
        """
        # Expect input shape: (B, C, H, W)
        # 1) Circular pad and 2) depthwise convolution (output spatial -> original H x W)
        x_padded = self.circ_pad(x)                   # (B, C, H+2, W+2)
        x_smooth = self.depthwise(x_padded)           # (B, C, H, W)

        # 3) Non-linearity
        x_act = self.selu(x_smooth)                   # (B, C, H, W)

        # 4) Downsample by factor of 2
        x_down = F.avg_pool2d(x_act, kernel_size=2)   # (B, C, H/2, W/2)

        # 5) Channel mixing at low resolution
        x_mix = self.pointwise(x_down)                # (B, C, H/2, W/2)

        # 6) Upsample back to original resolution
        x_up = F.interpolate(x_mix, scale_factor=2.0, mode='nearest')  # (B, C, H, W) if H and W are even

        # 7) Constant pad then crop the central region to demonstrate ConstantPad2d usage
        x_pad_const = self.const_pad(x_up)            # (B, C, H+2, W+2)
        # Crop back to original size by removing the constant border
        x_crop = x_pad_const[:, :, 1:-1, 1:-1]        # (B, C, H, W)

        # 8) Element-wise multiplicative combination with the original input
        merged = x * x_crop                           # (B, C, H, W)

        # 9) RMS normalization across channel dimension (keep dims)
        rms = torch.sqrt(torch.mean(merged ** 2, dim=1, keepdim=True) + self.eps)
        out = merged / rms                            # (B, C, H, W)

        return out

# Input generation helpers
def get_inputs() -> List[torch.Tensor]:
    """
    Returns a list with a single input tensor shaped (batch_size, in_channels, height, width).
    Values are randomly sampled from a standard normal distribution.
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs() -> List:
    """
    Returns the initialization arguments for the Model constructor:
    [in_channels, eps]
    """
    return [in_channels, eps_default]