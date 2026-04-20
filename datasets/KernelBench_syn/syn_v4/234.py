import torch
import torch.nn as nn

"""
Complex 3D upsampling module combining ConvTranspose3d, InstanceNorm3d, and RReLU.

Structure:
- Two staged transposed convolutional upsampling blocks (each: ConvTranspose3d -> InstanceNorm3d -> RReLU)
- A projected skip connection (single ConvTranspose3d) that upsamples the input directly to the final spatial size
- Residual fusion (element-wise addition) between the staged path and the skip projection
- Final 1x1x1 ConvTranspose3d projection to desired output channels
"""

# Configuration / default sizes
BATCH_SIZE = 2
IN_CHANNELS = 16
BASE_CHANNELS = 64
MID_CHANNELS = 32
OUT_CHANNELS = 3

DEPTH = 8   # input depth
HEIGHT = 8  # input height
WIDTH = 8   # input width

class Model(nn.Module):
    """
    3D upsampling/residual module.

    The network takes a low-resolution 3D feature volume and upsamples it by a factor of 4
    (two ConvTranspose3d stages each with stride=2). A parallel transposed convolution
    projects and upsamples the original input directly to the same final spatial dimensions,
    and the two paths are fused with an element-wise residual addition. Normalization and
    randomized leaky ReLU activations are applied between layers.
    """
    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        mid_channels: int,
        out_channels: int
    ):
        super(Model, self).__init__()
        # First upsampling stage: in -> base (upsample x2)
        self.up1 = nn.ConvTranspose3d(
            in_channels, base_channels,
            kernel_size=4, stride=2, padding=1
        )
        self.in1 = nn.InstanceNorm3d(base_channels)

        # Second upsampling stage: base -> mid (upsample x2, total x4)
        self.up2 = nn.ConvTranspose3d(
            base_channels, mid_channels,
            kernel_size=4, stride=2, padding=1
        )
        self.in2 = nn.InstanceNorm3d(mid_channels)

        # Skip path: directly project and upsample input by factor 4 to match spatial dims
        # kernel_size = stride = 4 with padding=0 yields output spatial size = 4 * input_size
        self.skip_proj = nn.ConvTranspose3d(
            in_channels, mid_channels,
            kernel_size=4, stride=4, padding=0
        )
        self.in_skip = nn.InstanceNorm3d(mid_channels)

        # Activation: Randomized Leaky ReLU
        self.act = nn.RReLU(lower=0.125, upper=0.333, inplace=False)

        # Final projection to desired output channels (1x1x1 conv transpose acts as conv)
        self.out_proj = nn.ConvTranspose3d(
            mid_channels, out_channels,
            kernel_size=1, stride=1, padding=0
        )
        self.in_final = nn.InstanceNorm3d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            Tensor of shape (batch_size, out_channels, D*4, H*4, W*4)
        """
        # Staged upsampling path
        y = self.up1(x)          # convtranspose up x2
        y = self.in1(y)          # instance norm
        y = self.act(y)          # non-linearity

        y = self.up2(y)          # convtranspose up x2 (total x4)
        y = self.in2(y)
        y = self.act(y)

        # Skip/projection path: direct upsample and channel project
        s = self.skip_proj(x)    # direct convtranspose up x4
        s = self.in_skip(s)
        s = self.act(s)

        # Residual fusion
        fused = y + s

        # Final projection and normalization
        out = self.out_proj(fused)
        out = self.in_final(out)
        out = self.act(out)

        return out

def get_inputs():
    """
    Returns a list containing the input tensor for the model.

    Shape: (BATCH_SIZE, IN_CHANNELS, DEPTH, HEIGHT, WIDTH)
    """
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, DEPTH, HEIGHT, WIDTH)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for the Model constructor.
    """
    return [IN_CHANNELS, BASE_CHANNELS, MID_CHANNELS, OUT_CHANNELS]