import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A 3D feature processing module that demonstrates a non-trivial interplay between
    ConvTranspose3d, MaxPool3d / MaxUnpool3d (with indices), and a Tanhshrink activation.

    Computation pattern:
      1. ConvTranspose3d to produce intermediate feature maps (keeps spatial dims).
      2. MaxPool3d (with indices) to downsample and collect pooling indices.
      3. Another ConvTranspose3d (1x1x1) to transform pooled features.
      4. MaxUnpool3d with the stored indices to restore spatial resolution.
      5. Residual addition with the pre-pooled features.
      6. Tanhshrink non-linearity and spatial aggregation to produce a compact output.

    The module returns a (batch_size, mid_channels) tensor by spatially averaging the
    activated feature maps after the unpool+residual stage.
    """
    def __init__(self, in_channels: int, mid_channels: int, pool_kernel: int = 2):
        """
        Args:
            in_channels (int): Number of input channels.
            mid_channels (int): Number of intermediate feature channels.
            pool_kernel (int): Kernel/stride for MaxPool3d / MaxUnpool3d (assumed equal).
        """
        super(Model, self).__init__()

        # First ConvTranspose3d: maintain spatial size (kernel_size=3, padding=1, stride=1)
        self.convt1 = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )

        # MaxPool3d will reduce each spatial dimension by 'pool_kernel'
        # return_indices=True to allow correct MaxUnpool3d restoration
        self.pool = nn.MaxPool3d(kernel_size=pool_kernel, stride=pool_kernel, return_indices=True)

        # A lightweight ConvTranspose3d to transform pooled features (1x1x1 conv-transpose)
        self.conv_mid = nn.ConvTranspose3d(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )

        # MaxUnpool3d to invert the pooling using the saved indices
        self.unpool = nn.MaxUnpool3d(kernel_size=pool_kernel, stride=pool_kernel)

        # Tanhshrink activation applied element-wise
        self.tanhshrink = nn.Tanhshrink()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the composed operations.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C_in, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (N, mid_channels) obtained by
                          spatially averaging the activated unpooled features.
        """
        # 1) Feature extraction / expansion (keeps spatial dimensions)
        y = self.convt1(x)  # shape: (N, mid_channels, D, H, W)

        # 2) Downsample and collect indices
        pooled, indices = self.pool(y)  # pooled shape: (N, mid_channels, D/2, H/2, W/2)

        # 3) Transform pooled features
        z = self.conv_mid(pooled)  # shape equal to pooled

        # 4) Unpool using the saved indices to restore spatial dimensions to y.size()
        # Provide output_size so unpool can place values correctly
        unpooled = self.unpool(z, indices, output_size=y.size())  # shape: same as y

        # 5) Residual connection: combine unpooled features with original conv features
        combined = unpooled + y

        # 6) Non-linearity and spatial aggregation
        activated = self.tanhshrink(combined)

        # Reduce spatial dims by averaging to return a compact per-channel descriptor
        N, C, D, H, W = activated.shape
        out = activated.view(N, C, -1).mean(dim=2)  # shape: (N, C)

        return out

# Configuration / default parameters at module level
batch_size = 4
in_channels = 3
mid_channels = 8
D = 16  # depth (must be divisible by pool kernel)
H = 16  # height
W = 16  # width
pool_kernel = 2

def get_inputs():
    """
    Produces a single input tensor compatible with the Model forward signature.

    Returns:
        list: [x] where x is of shape (batch_size, in_channels, D, H, W)
    """
    x = torch.randn(batch_size, in_channels, D, H, W)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters for the Model constructor.

    Returns:
        list: [in_channels, mid_channels, pool_kernel]
    """
    return [in_channels, mid_channels, pool_kernel]