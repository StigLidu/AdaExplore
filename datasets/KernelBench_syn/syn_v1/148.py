import torch
import torch.nn as nn

"""
Complex example combining LazyConvTranspose3d, ZeroPad2d and LazyBatchNorm2d.

Computation pattern:
1. 3D transposed convolution (learned) to upsample spatial dimensions.
2. Permute and collapse the depth dimension into the channel dimension to convert 3D output into a 2D feature map.
3. Zero-pad the 2D feature map.
4. Apply LazyBatchNorm2d (lazy initialization of num_features) to the padded 2D map.
5. Apply a non-linearity and a learned scalar affine transform (scale and bias).

Module-level configuration variables below describe input sizes and layer hyperparameters.
"""

# Configuration variables
batch_size = 4
in_channels = 3
depth = 5
height = 32
width = 32

# ConvTranspose3d hyperparameters (in_channels is lazy/inferred)
out_channels = 8
kernel_size = (3, 4, 4)
stride = (2, 2, 2)
padding = (1, 1, 1)
output_padding = (1, 0, 0)

# ZeroPad2d parameters (left, right, top, bottom)
pad2d = (1, 2, 1, 2)


class Model(nn.Module):
    """
    Model that converts a 5D tensor (N, C, D, H, W) into a 2D feature map via:
      - Lazy 3D transposed convolution to upsample spatial dims and produce feature channels
      - Collapse depth into channels to yield (N, C_out * D_out, H_out, W_out)
      - Zero padding in 2D
      - LazyBatchNorm2d (lazy-initialized num_features)
      - ReLU and a learned scalar affine transform

    The use of lazy modules allows the model to infer in_channels (for ConvTranspose3d)
    and num_features (for BatchNorm2d) on the first forward pass.
    """
    def __init__(
        self,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple,
        padding: tuple,
        output_padding: tuple,
        pad2d: tuple
    ):
        super(Model, self).__init__()
        # Lazy ConvTranspose3d: in_channels will be inferred when forward is called
        self.deconv3d = nn.LazyConvTranspose3d(
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding
        )

        # ZeroPad2d will pad the 2D maps after flattening depth into channels
        self.pad2d = nn.ZeroPad2d(pad2d)

        # LazyBatchNorm2d will infer num_features from input's channel dimension
        self.bn2d = nn.LazyBatchNorm2d()

        # Small learned affine transform applied after activation
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
          x: Tensor of shape (N, C_in, D, H, W)

        Returns:
          Tensor of shape (N, C_out * D_out, H_padded, W_padded)
        """
        # 1) 3D transposed convolution to upsample / increase channels
        y = self.deconv3d(x)  # shape: (N, out_channels, D_out, H_out, W_out)

        # 2) Move depth into channel dimension:
        #    (N, out_channels, D_out, H_out, W_out) -> (N, D_out, out_channels, H_out, W_out)
        y = y.permute(0, 2, 1, 3, 4)

        #    Collapse depth into channels:
        N, D_out, C_out, H_out, W_out = y.shape
        y = y.reshape(N, D_out * C_out, H_out, W_out)  # shape: (N, D_out*C_out, H_out, W_out)

        # 3) Zero padding in 2D
        y = self.pad2d(y)

        # 4) Lazy batch normalization in 2D (num_features inferred on first pass)
        y = self.bn2d(y)

        # 5) Non-linearity and learned scalar affine transform
        y = torch.relu(y)
        y = y * self.scale + self.bias

        return y


def get_inputs():
    """
    Returns a list containing a single input tensor shaped according to module-level config variables.
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]


def get_init_inputs():
    """
    Returns the initialization parameters for the Model constructor so the model can be instantiated externally.
    """
    return [out_channels, kernel_size, stride, padding, output_padding, pad2d]