import torch
import torch.nn as nn

"""
A more complex convolutional module that demonstrates a combination of Conv2d, GLU gating, and Mish activations.
Computation pattern:
    1. conv1: 3x3 convolution producing 2 * mid_channels so we can apply GLU along the channel dimension.
    2. GLU: split channels and gate to reduce to mid_channels.
    3. Mish activation.
    4. Depthwise convolution (groups=mid_channels) to mix spatial information per channel.
    5. Another Mish activation.
    6. Pointwise convolution (1x1) to project to out_channels.
    7. 1x1 skip projection from the input to out_channels and residual addition.
    8. Global average pooling to produce a (batch, out_channels) vector.

Module-level configuration variables control input/output shapes used by get_inputs.
"""

# Configuration
BATCH = 8
IN_CHANNELS = 3
MID_CHANNELS = 64
OUT_CHANNELS = 128
HEIGHT = 224
WIDTH = 224

class Model(nn.Module):
    """
    Convolutional block combining Conv2d, GLU gating, and Mish activations with a residual skip.

    Input:
        x: Tensor of shape (batch, in_channels, H, W)

    Output:
        Tensor of shape (batch, out_channels) after global average pooling
    """
    def __init__(self, in_channels: int = IN_CHANNELS, mid_channels: int = MID_CHANNELS, out_channels: int = OUT_CHANNELS):
        super(Model, self).__init__()
        # conv1 produces 2 * mid_channels so GLU can split and gate along channel dim (dim=1)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels * 2, kernel_size=3, padding=1, bias=True)
        self.glu = nn.GLU(dim=1)  # gates along channel dimension
        self.mish = nn.Mish()

        # depthwise convolution: groups=mid_channels (per-channel spatial mixing)
        self.depthwise = nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, padding=1, groups=mid_channels, bias=True)
        # pointwise projection to out_channels
        self.pointwise = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, bias=True)

        # skip connection projection to match out_channels for residual addition
        self.skip_proj = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=True)

        # global pooling to produce final vector
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor with shape (batch, in_channels, H, W)

        Returns:
            torch.Tensor: Output tensor with shape (batch, out_channels)
        """
        # 1) Conv -> produces 2*mid channels for GLU
        y = self.conv1(x)            # shape: (B, 2*mid, H, W)
        # 2) GLU gating reduces channels to mid_channels
        y = self.glu(y)              # shape: (B, mid, H, W)
        # 3) Non-linearity
        y = self.mish(y)             # shape: (B, mid, H, W)
        # 4) Depthwise conv per channel
        y = self.depthwise(y)        # shape: (B, mid, H, W)
        # 5) Another non-linearity
        y = self.mish(y)             # shape: (B, mid, H, W)
        # 6) Pointwise projection to out_channels
        y = self.pointwise(y)        # shape: (B, out_channels, H, W)

        # 7) Residual skip from input, projected to out_channels
        skip = self.skip_proj(x)     # shape: (B, out_channels, H, W)
        y = y + skip                 # residual addition

        # 8) Global average pooling to (B, out_channels, 1, 1) -> squeeze to (B, out_channels)
        y = self.global_pool(y)
        y = y.view(y.size(0), -1)
        return y

def get_inputs():
    """
    Returns:
        A list containing a single random input tensor consistent with the module-level configuration.
    """
    x = torch.randn(BATCH, IN_CHANNELS, HEIGHT, WIDTH)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for constructing the model externally if desired.
    The returned list follows the Model constructor argument order: (in_channels, mid_channels, out_channels)
    """
    return [IN_CHANNELS, MID_CHANNELS, OUT_CHANNELS]