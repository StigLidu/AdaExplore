import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex model combining Conv2d -> BatchNorm2d -> Hardshrink activation,
    channel-wise Dropout3d, 2D adaptive pooling to collapse height, 1D constant padding,
    a second Hardshrink, and a final linear projection.

    The model is designed to preserve the input width through convolution by using
    padding = kernel_size // 2, then pools height to 1 and keeps width the same so
    ConstantPad1d operates along the width dimension.
    """
    def __init__(
        self,
        in_channels: int,
        conv_out_channels: int,
        kernel_size: int,
        input_width: int,
        pad_left: int,
        pad_right: int,
        pad_value: float,
        hs_lambda1: float,
        hs_lambda2: float,
        dropout_p: float,
        linear_out_features: int,
    ):
        super(Model, self).__init__()
        # Convolution that keeps spatial dimensions (same padding)
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, conv_out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(conv_out_channels)

        # Non-linear shrink activations (two different thresholds)
        self.hs1 = nn.Hardshrink(lambd=hs_lambda1)
        self.hs2 = nn.Hardshrink(lambd=hs_lambda2)

        # Dropout that zeros entire channels (applied on a 5D tensor, so we unsqueeze a depth dim)
        self.drop3d = nn.Dropout3d(p=dropout_p)

        # Pooling to collapse the height dimension to 1 while keeping width the same
        # AdaptiveAvgPool2d requires a fixed output size, so we set it to (1, input_width)
        self.pool = nn.AdaptiveAvgPool2d((1, input_width))

        # ConstantPad1d will pad the width dimension (input shape will be N x C x L)
        self.pad1d = nn.ConstantPad1d((pad_left, pad_right), pad_value)

        # Final linear projection: flattened channels * padded_width -> linear_out_features
        padded_width = input_width + pad_left + pad_right
        self.linear = nn.Linear(conv_out_channels * padded_width, linear_out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
         - Conv2d -> BatchNorm2d
         - Hardshrink activation (elementwise)
         - Unsqueeze depth dim -> Dropout3d -> Squeeze back
         - AdaptiveAvgPool2d to (1, input_width) -> squeeze height to create (N, C, W)
         - ConstantPad1d along width -> second Hardshrink
         - Flatten and Linear projection

        Args:
            x (torch.Tensor): Input tensor of shape (N, in_channels, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (N, linear_out_features)
        """
        # Convolution + normalization
        x = self.conv(x)
        x = self.bn(x)

        # Elementwise shrink activation
        x = self.hs1(x)

        # Make it 5D (N, C, D, H, W) with a singleton depth so Dropout3d can drop whole channels
        x = x.unsqueeze(2)          # shape: (N, C, 1, H, W)
        x = self.drop3d(x)
        x = x.squeeze(2)            # back to (N, C, H, W)

        # Pool height to 1 while preserving width
        x = self.pool(x)            # shape: (N, C, 1, W)
        x = x.squeeze(2)            # shape: (N, C, W)

        # Pad width dimension and apply a second shrink activation
        x = self.pad1d(x)           # shape: (N, C, W + pad_left + pad_right)
        x = self.hs2(x)

        # Flatten per example and project
        x = x.flatten(1)            # shape: (N, C * padded_width)
        x = self.linear(x)          # shape: (N, linear_out_features)
        return x

# Configuration variables
batch_size = 8
in_channels = 3
conv_out_channels = 16
kernel_size = 3
input_height = 32
input_width = 64
pad_left = 2
pad_right = 3
pad_value = 0.1
hs_lambda1 = 0.5
hs_lambda2 = 1.0
dropout_p = 0.2
linear_out_features = 128

def get_inputs():
    """
    Returns a list containing a single input tensor shaped according to configuration.
    Shape: (batch_size, in_channels, input_height, input_width)
    """
    x = torch.randn(batch_size, in_channels, input_height, input_width)
    return [x]

def get_init_inputs():
    """
    Returns the initialization arguments for Model in the same order as the constructor:
    in_channels, conv_out_channels, kernel_size, input_width, pad_left, pad_right,
    pad_value, hs_lambda1, hs_lambda2, dropout_p, linear_out_features
    """
    return [
        in_channels,
        conv_out_channels,
        kernel_size,
        input_width,
        pad_left,
        pad_right,
        pad_value,
        hs_lambda1,
        hs_lambda2,
        dropout_p,
        linear_out_features,
    ]