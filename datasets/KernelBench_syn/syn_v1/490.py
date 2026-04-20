import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex module combining PixelUnshuffle, MaxPool2d/MaxUnpool2d with indices,
    a Sigmoid non-linearity, and a 1x1 convolution to fuse channels.

    Computation pipeline:
    1. PixelUnshuffle: rearrange spatial -> channel (downscale spatial by factor r).
    2. MaxPool2d (with return_indices=True): spatial pooling that produces indices.
    3. Sigmoid activation on pooled features.
    4. MaxUnpool2d: invert the pooling using indices to restore the pre-pooled spatial size.
    5. 1x1 Conv2d: reduce expanded channels back to original input channel count.

    This creates an interesting interplay between spatial reorganization and pooling/unpooling
    while using a non-linear gating (Sigmoid) in the pooled domain.
    """
    def __init__(
        self,
        in_channels: int,
        downscale_factor: int = 2,
        pool_kernel: int = 2,
        pool_stride: int = 2
    ):
        """
        Args:
            in_channels (int): Number of channels in the input tensor.
            downscale_factor (int): PixelUnshuffle downscale factor (must divide H and W).
            pool_kernel (int): Kernel size for MaxPool2d / MaxUnpool2d.
            pool_stride (int): Stride for MaxPool2d / MaxUnpool2d.
        """
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.downscale = downscale_factor

        # After PixelUnshuffle the channel count becomes in_channels * downscale^2
        expanded_channels = in_channels * (downscale_factor ** 2)

        # Layers
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor)
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride, return_indices=True)
        self.sigmoid = nn.Sigmoid()
        self.unpool = nn.MaxUnpool2d(kernel_size=pool_kernel, stride=pool_stride)
        # 1x1 conv to bring channels back to original channel count after unpool
        self.conv1x1 = nn.Conv2d(in_channels=expanded_channels, out_channels=in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_channels, H, W).
                              H and W must be divisible by downscale_factor and pool strides.

        Returns:
            torch.Tensor: Tensor of shape (batch, in_channels, H/downscale_factor, W/downscale_factor)
                          after unpooling and channel fusion (spatial is restored to the PixelUnshuffle output size).
        """
        # 1) Rearrange spatial resolution into channels
        unshuffled = self.pixel_unshuffle(x)
        # 2) Max pool with indices
        pooled, indices = self.pool(unshuffled)
        # 3) Non-linear gating
        gated = self.sigmoid(pooled)
        # 4) Unpool using the saved indices to get back to the pre-pooled spatial resolution
        # Provide output_size to ensure exact shape match with unshuffled
        unpooled = self.unpool(gated, indices, output_size=unshuffled.size())
        # 5) Fuse expanded channels back to original channels
        out = self.conv1x1(unpooled)
        return out

# Configuration variables
batch_size = 4
in_channels = 3
height = 32
width = 32
downscale_factor = 2  # PixelUnshuffle downscale factor (must divide height and width)
pool_kernel = 2
pool_stride = 2

def get_inputs():
    """
    Returns example input tensor matching the configuration above.
    Ensure height and width are divisible by downscale_factor and pool operations.
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for the Model constructor.
    Order: in_channels, downscale_factor, pool_kernel, pool_stride
    """
    return [in_channels, downscale_factor, pool_kernel, pool_stride]