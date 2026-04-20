import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex 2D image processing model that demonstrates a small encoder-decoder
    pattern using LazyConv2d (lazy-initialized conv), a downsampling MaxPool2d,
    and two ConvTranspose2d layers to upsample back to the original resolution.

    Computation graph:
      x -> LazyConv2d (preserve HxW) -> ReLU ->
           MaxPool2d (downsample by 2) ->
           ConvTranspose2d (upsample by 2) -> ReLU ->
           ConvTranspose2d (channel restore, preserve HxW) -> Sigmoid
    """
    def __init__(self,
                 mid_channels: int = 48,
                 bottleneck_channels: int = 24,
                 out_channels: int = 3):
        """
        Args:
            mid_channels: number of output channels from the lazy conv (encoder width)
            bottleneck_channels: intermediate channel count in decoder
            out_channels: final number of channels to produce (e.g., RGB = 3)
        """
        super(Model, self).__init__()

        # LazyConv2d will infer in_channels from the first forward pass input
        # Kernel size 3 with padding keeps spatial resolution the same.
        self.encoder_conv = nn.LazyConv2d(out_channels=mid_channels, kernel_size=3, padding=1)

        # MaxPool2d reduces spatial dimensions by a factor of 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # First transpose conv upsamples by factor 2 to invert the pooling.
        # kernel_size=2, stride=2 is a common choice to double spatial dims.
        self.upconv1 = nn.ConvTranspose2d(in_channels=mid_channels,
                                          out_channels=bottleneck_channels,
                                          kernel_size=2,
                                          stride=2)

        # Final transpose conv restores the desired number of output channels
        # while preserving spatial dims (kernel_size=3, padding=1).
        self.upconv2 = nn.ConvTranspose2d(in_channels=bottleneck_channels,
                                          out_channels=out_channels,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1)

        # Optional small parameter for numerical stability in activations (kept as attribute)
        self.eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder-decoder.

        Args:
            x: input tensor with shape (B, C_in, H, W)

        Returns:
            tensor with shape (B, out_channels, H, W) with values in (0,1) via sigmoid
        """
        # Encoder conv (lazy) followed by non-linearity
        z = self.encoder_conv(x)           # (B, mid_channels, H, W)
        z = F.relu(z, inplace=True)

        # Spatial downsample
        z = self.pool(z)                   # (B, mid_channels, H/2, W/2)

        # Decoder upsamples back to original resolution
        z = self.upconv1(z)                # (B, bottleneck_channels, H, W)
        z = F.relu(z, inplace=True)

        # Final channel-mapping transpose conv
        z = self.upconv2(z)                # (B, out_channels, H, W)

        # Bound outputs to (0,1) with sigmoid for image-like outputs
        out = torch.sigmoid(z + self.eps)

        return out

# Configuration for synthetic inputs
batch_size = 8
in_channels = 3
height = 128  # even to ensure pooling/upconv symmetry
width = 64    # even to ensure pooling/upconv symmetry

def get_inputs():
    """
    Returns a list containing one input tensor matching the expected input shape:
    (batch_size, in_channels, height, width).
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization inputs if needed. The LazyConv2d does not require
    explicit initialization inputs; return empty list to conform to examples.
    """
    return []