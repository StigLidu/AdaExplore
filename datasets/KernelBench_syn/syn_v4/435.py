import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Upsampling residual block that combines transposed convolutions and Alpha Dropout.
    The module performs two spatial upsampling steps using ConvTranspose2d, applies an
    intermediate 2D convolution for feature mixing, uses AlphaDropout for regularization,
    and adds a learned residual skip connection (1x1 conv) from the input after interpolation.
    """
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        dropout_p: float = 0.1,
    ):
        """
        Initializes the upsampling residual block.

        Args:
            in_channels (int): Number of input channels.
            mid_channels (int): Number of channels in intermediate feature maps.
            out_channels (int): Number of output channels after upsampling.
            kernel_size (int, optional): Kernel size for ConvTranspose2d layers. Defaults to 4.
            stride (int, optional): Stride for ConvTranspose2d layers. Defaults to 2.
            padding (int, optional): Padding for ConvTranspose2d layers. Defaults to 1.
            dropout_p (float, optional): Probability for AlphaDropout. Defaults to 0.1.
        """
        super(Model, self).__init__()

        # First upsampling stage: doubles spatial dimensions
        self.up1 = nn.ConvTranspose2d(
            in_channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=padding
        )

        # Feature mixing (keeps spatial dims)
        self.mix_conv = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)

        # Regularization
        self.alpha_dropout = nn.AlphaDropout(p=dropout_p)

        # Second upsampling stage: doubles spatial dimensions again
        self.up2 = nn.ConvTranspose2d(
            mid_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
        )

        # Learned skip projection to match channels for residual addition
        self.skip_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Simple activation
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the upsampling residual block.

        Pipeline:
            x -> up1 (ConvTranspose2d) -> ReLU
             -> AlphaDropout -> mix_conv -> ReLU
             -> up2 (ConvTranspose2d)
            residual: interpolate(x) -> skip_proj -> add to up2 output
            -> final ReLU

        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H * 4, W * 4)
        """
        # First upsample
        u = self.up1(x)
        u = self.act(u)

        # Regularize with AlphaDropout (keeps self-normalizing networks properties if needed)
        u = self.alpha_dropout(u)

        # Mix features spatially without changing resolution
        u = self.mix_conv(u)
        u = self.act(u)

        # Second upsample
        out = self.up2(u)

        # Prepare residual: interpolate input to match spatial size and project channels
        target_h, target_w = out.shape[-2], out.shape[-1]
        skip = F.interpolate(x, size=(target_h, target_w), mode='nearest')
        skip = self.skip_proj(skip)

        # Residual addition and final activation
        out = out + skip
        out = self.act(out)
        return out

# Configuration (module-level)
batch_size = 8
in_channels = 3
mid_channels = 64
out_channels = 3
input_height = 32
input_width = 32
kernel_size = 4
stride = 2
padding = 1
dropout_p = 0.15

def get_inputs():
    """
    Returns:
        list: single-element list containing a randomly initialized input tensor
              of shape (batch_size, in_channels, input_height, input_width).
    """
    x = torch.randn(batch_size, in_channels, input_height, input_width)
    return [x]

def get_init_inputs():
    """
    Returns:
        list: initialization arguments for the Model constructor in order.
    """
    return [in_channels, mid_channels, out_channels, kernel_size, stride, padding, dropout_p]