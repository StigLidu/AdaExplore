import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A moderately complex 3D encoder-decoder style module that demonstrates:
    - Lazy 3D convolutions (nn.LazyConv3d) which infer input channels at first forward
    - Max pooling with indices (nn.MaxPool3d) and corresponding MaxUnpool3d to invert pooling
    - Spatial upsampling (nn.Upsample) after unpooling and skip-connection fusion

    The network performs two levels of downsampling via MaxPool3d, processes a small bottleneck,
    then performs hierarchical unpooling to recover spatial resolution, fuses a skip connection,
    upsamples, and projects to a single-channel output.
    """
    def __init__(self, mid_channels: int = 16, bottleneck_channels: int = 32, upsample_scale: int = 2):
        """
        Initializes convolutional, pooling, unpooling and upsampling layers.

        Args:
            mid_channels (int): Number of channels used in the encoder/decoder intermediate layers.
            bottleneck_channels (int): Number of channels in the bottleneck convolution.
            upsample_scale (int): Multiplicative scale factor for the final Upsample layer.
        """
        super(Model, self).__init__()

        # Encoder convolutions (lazy so in_channels is inferred at first forward)
        self.conv1 = nn.LazyConv3d(out_channels=mid_channels, kernel_size=3, padding=1)
        self.conv2 = nn.LazyConv3d(out_channels=bottleneck_channels, kernel_size=3, padding=1)

        # Bottleneck processing
        self.conv3 = nn.LazyConv3d(out_channels=bottleneck_channels, kernel_size=3, padding=1)

        # Decoder / projection convolutions
        self.conv4 = nn.LazyConv3d(out_channels=mid_channels, kernel_size=3, padding=1)
        self.final_conv = nn.LazyConv3d(out_channels=1, kernel_size=1)

        # Pooling and corresponding unpooling (kernel/stride chosen to downsample by 2)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.unpool2 = nn.MaxUnpool3d(kernel_size=2, stride=2)
        self.unpool1 = nn.MaxUnpool3d(kernel_size=2, stride=2)

        # Non-linearity and upsampling
        self.relu = nn.ReLU(inplace=True)
        # Trilinear upsampling to increase spatial resolution after fusion
        self.upsample = nn.Upsample(scale_factor=upsample_scale, mode='trilinear', align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder-decoder pipeline.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, D, H, W)

        Returns:
            torch.Tensor: Single-channel output tensor with upsampled spatial resolution.
        """
        # Encoder level 1
        skip = self.relu(self.conv1(x))          # shape -> (B, mid_channels, D, H, W)
        size_skip = skip.size()                  # used for corresponding unpool output_size
        pooled1, idx1 = self.pool1(skip)         # downsampled by 2

        # Encoder level 2
        e2 = self.relu(self.conv2(pooled1))      # (B, bottleneck_channels, D/2, H/2, W/2)
        size_e2 = e2.size()
        pooled2, idx2 = self.pool2(e2)           # further downsampled by 2

        # Bottleneck
        bottleneck = self.relu(self.conv3(pooled2))

        # Decoder: unpool the second level back to e2 size
        u2 = self.unpool2(bottleneck, idx2, output_size=size_e2)
        u2 = self.relu(u2)

        # Optionally process after first unpool
        u2 = self.relu(self.conv4(u2))

        # Unpool to recover skip size
        u1 = self.unpool1(u2, idx1, output_size=size_skip)
        u1 = self.relu(u1)

        # Fuse with encoder skip connection (element-wise addition)
        fused = u1 + skip

        # Upsample spatially and project to final single-channel output
        up = self.upsample(fused)
        out = self.final_conv(up)

        return out

# Configuration / initialization parameters
batch_size = 2
in_channels = 3
depth = 32
height = 32
width = 32

mid_channels = 16
bottleneck_channels = 32
upsample_scale = 2

def get_inputs():
    """
    Returns a list containing the example input tensor for the model.
    Shape: (batch_size, in_channels, depth, height, width)
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Returns the initialization arguments for Model.__init__ in a list.
    """
    return [mid_channels, bottleneck_channels, upsample_scale]