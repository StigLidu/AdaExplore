import torch
import torch.nn as nn
from typing import Tuple, List, Any

class Model(nn.Module):
    """
    Model that combines a 3D transposed convolution, Softplus non-linearity,
    a depth-to-channel re-arrangement, 2D nearest-neighbor upsampling, and
    a learned positive scaling computed from spatial statistics.

    Computation steps:
    1. ConvTranspose3d to expand spatial/temporal dimensions.
    2. Softplus activation.
    3. Collapse the depth dimension into the channel dimension to produce a 4D tensor.
    4. UpsamplingNearest2d to upsample height/width.
    5. Compute spatial global mean per-channel, pass through Softplus to ensure positivity,
       and scale feature maps channelwise by this factor.
    """
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        kernel_size_3d: Tuple[int, int, int],
        stride_3d: Tuple[int, int, int],
        padding_3d: Tuple[int, int, int],
        output_padding_3d: Tuple[int, int, int],
        up_scale: int,
    ):
        """
        Initializes the model components.

        Args:
            in_channels (int): Number of input channels for the ConvTranspose3d.
            mid_channels (int): Number of output channels produced by ConvTranspose3d.
            kernel_size_3d (tuple): 3D kernel size for ConvTranspose3d.
            stride_3d (tuple): 3D stride for ConvTranspose3d.
            padding_3d (tuple): 3D padding for ConvTranspose3d.
            output_padding_3d (tuple): 3D output padding for ConvTranspose3d.
            up_scale (int): Integer upsampling factor for height and width with UpsamplingNearest2d.
        """
        super(Model, self).__init__()
        self.deconv3d = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size_3d,
            stride=stride_3d,
            padding=padding_3d,
            output_padding=output_padding_3d,
            bias=True,
        )
        self.softplus = nn.Softplus()
        self.upsample = nn.UpsamplingNearest2d(scale_factor=up_scale)

        # Keep parameters for informational/debugging uses
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.up_scale = up_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch, mid_channels * depth_out, height_out * up_scale, width_out * up_scale)
        """
        # 1) 3D transposed convolution: (B, C_in, D, H, W) -> (B, C_mid, D2, H2, W2)
        x = self.deconv3d(x)

        # 2) Non-linear activation
        x = self.softplus(x)

        # 3) Collapse depth into channel dimension to prepare for 2D ops:
        #    (B, C_mid, D2, H2, W2) -> (B, C_mid * D2, H2, W2)
        B, C_mid, D2, H2, W2 = x.size()
        # Ensure contiguous before view to avoid unexpected behavior
        x = x.contiguous().view(B, C_mid * D2, H2, W2)

        # 4) 2D nearest neighbor upsampling on height/width
        x = self.upsample(x)

        # 5) Compute per-channel spatial mean, ensure positive scaling via Softplus,
        #    and apply channel-wise multiplicative gating.
        #    mean over spatial dims -> shape (B, C', 1, 1)
        spatial_mean = x.mean(dim=(2, 3), keepdim=True)
        scale = self.softplus(spatial_mean)  # positive scaling factor
        x = x * scale  # broadcast multiplication

        return x


# Configuration / initialization parameters
batch_size = 8
in_channels = 16
mid_channels = 24
depth = 4
height = 16
width = 16

# ConvTranspose3d parameters (3D deconvolution)
kernel_size_3d = (3, 3, 3)
stride_3d = (2, 2, 2)
padding_3d = (1, 1, 1)
output_padding_3d = (1, 1, 1)

# Upsampling scale factor for height/width
up_scale = 3

def get_inputs() -> List[torch.Tensor]:
    """
    Produces a list with one input tensor for the model.

    Returns:
        List[torch.Tensor]: [x] where x has shape (batch_size, in_channels, depth, height, width)
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs() -> List[Any]:
    """
    Returns the initialization parameters for constructing the Model instance.

    Returns:
        List containing the tuple of initialization arguments in the same order as Model.__init__.
    """
    return [
        in_channels,
        mid_channels,
        kernel_size_3d,
        stride_3d,
        padding_3d,
        output_padding_3d,
        up_scale,
    ]