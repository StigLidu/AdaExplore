import torch
import torch.nn as nn
from typing import List, Any, Tuple

class Model(nn.Module):
    """
    Complex composite module that:
    - Applies a 3D transposed convolution to expand spatial dimensions.
    - Takes the central depth slice and applies a constant 2D padding.
    - Flattens spatial dimensions into a 1D signal per channel and applies adaptive 1D max pooling.
    - Performs a simple L1-style normalization over the pooled sequence.

    Input shape: (batch_size, in_channels, D, H, W)
    Output shape: (batch_size, out_channels, pool_output_size)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        deconv_kernel: Tuple[int, int, int],
        deconv_stride: Tuple[int, int, int],
        deconv_padding: Tuple[int, int, int],
        deconv_output_padding: Tuple[int, int, int],
        pad_2d: Tuple[int, int, int, int],
        pad_value: float,
        pool_output_size: int
    ):
        """
        Initializes internal layers.

        Args:
            in_channels: Number of input channels for ConvTranspose3d.
            out_channels: Number of output channels for ConvTranspose3d.
            deconv_kernel: Kernel size for ConvTranspose3d (D, H, W).
            deconv_stride: Stride for ConvTranspose3d (D, H, W).
            deconv_padding: Padding for ConvTranspose3d (D, H, W).
            deconv_output_padding: Output padding for ConvTranspose3d (D, H, W).
            pad_2d: ConstantPad2d padding tuple (left, right, top, bottom).
            pad_value: Constant value to use for padding.
            pool_output_size: Output size for AdaptiveMaxPool1d.
        """
        super(Model, self).__init__()
        # 3D transposed convolution to expand spatial dimensions and channels
        self.deconv3d = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=deconv_kernel,
            stride=deconv_stride,
            padding=deconv_padding,
            output_padding=deconv_output_padding,
            bias=True
        )
        # Constant 2D padding layer to be applied to a depth slice
        self.pad2d = nn.ConstantPad2d(pad_2d, pad_value)
        # Adaptive 1D max pooling applied to flattened spatial sequence per channel
        self.pool1d = nn.AdaptiveMaxPool1d(pool_output_size)
        # Small epsilon for numerical stability in normalization
        self.eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Steps:
        1. Apply ConvTranspose3d -> (B, out_channels, D2, H2, W2)
        2. Apply ReLU activation.
        3. Extract the central depth slice across D2 -> (B, out_channels, H2, W2)
        4. Apply ConstantPad2d to the 2D slice -> (B, out_channels, H_p, W_p)
        5. Flatten spatial dims to a 1D sequence per channel -> (B, out_channels, L)
        6. Apply AdaptiveMaxPool1d -> (B, out_channels, pool_output_size)
        7. Normalize each (B, out_channels, :) by its L1 norm across the last dim.

        Args:
            x: Input tensor of shape (B, in_channels, D, H, W)

        Returns:
            Tensor of shape (B, out_channels, pool_output_size)
        """
        # 1 -> 2
        y = self.deconv3d(x)
        y = torch.relu(y)

        # 3: select the central depth slice
        # handle dynamic depth
        D2 = y.shape[2]
        center_idx = D2 // 2
        slice2d = y[:, :, center_idx, :, :]  # shape: (B, out_channels, H2, W2)

        # 4: constant 2D padding
        padded = self.pad2d(slice2d)  # shape: (B, out_channels, H_p, W_p)

        # 5: flatten spatial dims into a sequence (length L)
        B, C, H_p, W_p = padded.shape
        seq = padded.view(B, C, H_p * W_p)  # shape: (B, C, L)

        # 6: adaptive max pool along the sequence dimension
        pooled = self.pool1d(seq)  # shape: (B, C, pool_output_size)

        # 7: L1-style normalization along the pooled sequence dimension
        denom = pooled.abs().sum(dim=2, keepdim=True).clamp(min=self.eps)
        normalized = pooled / denom

        return normalized

# Configuration variables
batch_size = 8
in_channels = 3
out_channels = 16
depth = 6
height = 32
width = 32

# ConvTranspose3d parameters
deconv_kernel = (3, 5, 5)
deconv_stride = (2, 2, 2)
deconv_padding = (1, 2, 2)
deconv_output_padding = (1, 1, 1)

# ConstantPad2d parameters (left, right, top, bottom)
pad_2d = (2, 2, 3, 3)
pad_value = 0.1

# AdaptiveMaxPool1d output size
pool_output_size = 64

def get_inputs() -> List[torch.Tensor]:
    """
    Returns a list containing the input tensor for the model.

    Input shape: (batch_size, in_channels, depth, height, width)
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs() -> List[Any]:
    """
    Returns the initialization parameters for the Model constructor so the module can be instantiated externally.
    """
    return [
        in_channels,
        out_channels,
        deconv_kernel,
        deconv_stride,
        deconv_padding,
        deconv_output_padding,
        pad_2d,
        pad_value,
        pool_output_size
    ]