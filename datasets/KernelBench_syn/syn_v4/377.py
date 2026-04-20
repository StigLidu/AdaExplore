import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Image-context fusion model that:
    - Applies a 2D convolution to an input image
    - Uses adaptive average pooling to get a compact image embedding
    - Fuses the image embedding with a separate context vector via a Bilinear layer
    - Applies a LogSigmoid activation to the fused embedding

    This demonstrates combining Conv2d, AdaptiveAvgPool2d, Bilinear and LogSigmoid
    into a compact, multi-input forward pass.
    """
    def __init__(
        self,
        in_channels: int,
        conv_out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        context_dim: int = 32,
        bilinear_out_features: int = 10
    ):
        """
        Initializes layers and parameters.

        Args:
            in_channels (int): Number of channels in the input image.
            conv_out_channels (int): Number of output channels produced by the convolution.
            kernel_size (int): Size of the convolution kernel.
            stride (int): Stride of the convolution.
            padding (int): Zero-padding added to both sides of the input.
            context_dim (int): Dimensionality of the separate context vector.
            bilinear_out_features (int): Output features from the bilinear fusion.
        """
        super(Model, self).__init__()
        # 2D convolution to extract spatial features
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=conv_out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        # Non-linearity
        self.relu = nn.ReLU(inplace=True)
        # Pool to a single spatial location per channel to get a channel descriptor
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # Bilinear fusion: combines image descriptor with context vector
        self.bilinear = nn.Bilinear(conv_out_channels, context_dim, bilinear_out_features)
        # Final activation
        self.logsigmoid = nn.LogSigmoid()

    def forward(self, x_img: torch.Tensor, x_ctx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining image and context.

        Args:
            x_img (torch.Tensor): Image tensor of shape (batch_size, in_channels, H, W).
            x_ctx (torch.Tensor): Context tensor of shape (batch_size, context_dim).

        Returns:
            torch.Tensor: Fused output of shape (batch_size, bilinear_out_features) with LogSigmoid applied.
        """
        # Convolution + ReLU
        feat = self.conv(x_img)            # (batch, conv_out_channels, H', W')
        feat = self.relu(feat)             # same shape

        # Global channel descriptor via adaptive pooling
        pooled = self.pool(feat)           # (batch, conv_out_channels, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)  # (batch, conv_out_channels)

        # Bilinear fusion with context vector
        fused = self.bilinear(pooled, x_ctx)  # (batch, bilinear_out_features)

        # Non-linear squashing
        out = self.logsigmoid(fused)          # (batch, bilinear_out_features)
        return out

# Configuration / default parameters for the example
batch_size = 8
in_channels = 3
conv_out_channels = 64
height = 128
width = 128
kernel_size = 3
stride = 1
padding = 1
context_dim = 32
bilinear_out_features = 16

def get_inputs():
    """
    Returns example inputs:
    - x_img: random image tensor of shape (batch_size, in_channels, height, width)
    - x_ctx: random context tensor of shape (batch_size, context_dim)
    """
    x_img = torch.randn(batch_size, in_channels, height, width)
    x_ctx = torch.randn(batch_size, context_dim)
    return [x_img, x_ctx]

def get_init_inputs():
    """
    Returns the initialization parameters for Model in the same order as the constructor:
    [in_channels, conv_out_channels, kernel_size, stride, padding, context_dim, bilinear_out_features]
    """
    return [in_channels, conv_out_channels, kernel_size, stride, padding, context_dim, bilinear_out_features]