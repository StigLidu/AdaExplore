import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A 3D upsampling-and-refinement module that uses two ConvTranspose3d layers
    interleaved with LazyInstanceNorm3d and ReLU, finishing with a 3D max pooling.
    This pattern demonstrates learned upsampling (ConvTranspose3d), channel-wise
    adaptive normalization (LazyInstanceNorm3d), non-linearity (ReLU), and a
    final spatial downsampling (MaxPool3d) to produce a refined volumetric output.
    """
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel1,
        stride1,
        padding1,
        output_padding1,
        kernel2,
        stride2,
        padding2,
        output_padding2,
        pool_kernel,
        pool_stride,
        pool_padding,
    ):
        super(Model, self).__init__()
        # First transposed conv upsamples the spatial dims
        self.upconv1 = nn.ConvTranspose3d(
            in_channels, mid_channels,
            kernel_size=kernel1,
            stride=stride1,
            padding=padding1,
            output_padding=output_padding1,
            bias=False
        )
        # LazyInstanceNorm3d will infer num_features from the first input it sees
        self.norm1 = nn.LazyInstanceNorm3d()
        self.act = nn.ReLU(inplace=True)

        # Second transposed conv refines features without changing spatial resolution
        self.upconv2 = nn.ConvTranspose3d(
            mid_channels, out_channels,
            kernel_size=kernel2,
            stride=stride2,
            padding=padding2,
            output_padding=output_padding2,
            bias=False
        )
        # Another lazy instance normalization after refinement
        self.norm2 = nn.LazyInstanceNorm3d()

        # Final 3D max pooling to reduce spatial resolution and emphasize strong responses
        self.pool = nn.MaxPool3d(kernel_size=pool_kernel, stride=pool_stride, padding=pool_padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        1. Upsample input with ConvTranspose3d
        2. Apply InstanceNorm and ReLU
        3. Refine with a second ConvTranspose3d
        4. Apply second InstanceNorm and ReLU
        5. Apply MaxPool3d to obtain the final output

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W)

        Returns:
            torch.Tensor: Output tensor after upsampling, refinement, normalization, activation, and pooling.
        """
        x = self.upconv1(x)   # learned upsampling
        x = self.norm1(x)     # channel-wise adaptive normalization (lazy)
        x = self.act(x)       # non-linearity

        x = self.upconv2(x)   # refinement conv
        x = self.norm2(x)     # normalize refined features
        x = self.act(x)       # non-linearity

        x = self.pool(x)      # spatial max-pooling
        return x

# Module-level configuration (example sizes)
BATCH_SIZE = 2
IN_CHANNELS = 8
MID_CHANNELS = 16
OUT_CHANNELS = 4

# Input volumetric dimensions
DEPTH = 8
HEIGHT = 16
WIDTH = 16

# ConvTranspose3d layer 1 parameters (upsample by factor 2)
KERNEL1 = (4, 4, 4)
STRIDE1 = (2, 2, 2)
PADDING1 = (1, 1, 1)
OUTPUT_PADDING1 = (0, 0, 0)

# ConvTranspose3d layer 2 parameters (keep spatial dims)
KERNEL2 = (3, 3, 3)
STRIDE2 = (1, 1, 1)
PADDING2 = (1, 1, 1)
OUTPUT_PADDING2 = (0, 0, 0)

# MaxPool3d parameters (downsample by factor 2)
POOL_KERNEL = (2, 2, 2)
POOL_STRIDE = (2, 2, 2)
POOL_PADDING = (0, 0, 0)

def get_inputs():
    """
    Generates a random volumetric input tensor for testing.

    Returns:
        list: A single-element list containing the input tensor with shape
              (BATCH_SIZE, IN_CHANNELS, DEPTH, HEIGHT, WIDTH).
    """
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, DEPTH, HEIGHT, WIDTH)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters for the Model constructor in the order
    expected by __init__.

    Returns:
        list: Initialization parameters for Model.
    """
    return [
        IN_CHANNELS,
        MID_CHANNELS,
        OUT_CHANNELS,
        KERNEL1,
        STRIDE1,
        PADDING1,
        OUTPUT_PADDING1,
        KERNEL2,
        STRIDE2,
        PADDING2,
        OUTPUT_PADDING2,
        POOL_KERNEL,
        POOL_STRIDE,
        POOL_PADDING,
    ]