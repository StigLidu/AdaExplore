import torch
import torch.nn as nn

# Configuration / default sizes
batch_size = 8
in_channels = 3
mid_channels = 16
out_channels = 32
height = 64
width = 64

# Convolution parameters
conv1_kernel = 3
conv1_stride = 1
conv1_padding = 1

conv2_kernel = 3
conv2_stride = 1
conv2_padding = 1

# AvgPool2d parameters
pool_kernel = 2
pool_stride = 2
pool_padding = 0

# Final pointwise conv (1x1)
conv3_kernel = 1
conv3_stride = 1
conv3_padding = 0


class Model(nn.Module):
    """
    A moderately complex model that combines Conv2d, AvgPool2d and LazyInstanceNorm1d.
    Computation graph (high level):
      x -> conv1 -> relu -> conv2 -> avgpool2d -> (N, C, H', W')
         -> reshape to (N, C, L) -> LazyInstanceNorm1d -> reshape back -> conv3 (1x1)
         -> residual merge with upsampled input projection -> final activation
    This demonstrates spatial conv processing, pooling, channel-wise instance normalization
    applied on a flattened spatial sequence, and a pointwise conv + residual-style merge.
    """
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        conv1_kernel: int = 3,
        conv1_stride: int = 1,
        conv1_padding: int = 1,
        conv2_kernel: int = 3,
        conv2_stride: int = 1,
        conv2_padding: int = 1,
        pool_kernel: int = 2,
        pool_stride: int = 2,
        pool_padding: int = 0,
        conv3_kernel: int = 1,
        conv3_stride: int = 1,
        conv3_padding: int = 0,
    ):
        super(Model, self).__init__()
        # First spatial feature extractor
        self.conv1 = nn.Conv2d(
            in_channels,
            mid_channels,
            kernel_size=conv1_kernel,
            stride=conv1_stride,
            padding=conv1_padding,
            bias=True,
        )
        # Second conv to increase channels before normalization
        self.conv2 = nn.Conv2d(
            mid_channels,
            out_channels,
            kernel_size=conv2_kernel,
            stride=conv2_stride,
            padding=conv2_padding,
            bias=True,
        )
        # Pooling to reduce spatial dimensions
        self.avg_pool = nn.AvgPool2d(
            kernel_size=pool_kernel, stride=pool_stride, padding=pool_padding
        )
        # LazyInstanceNorm1d will be initialized on the first forward call, expects input (N, C, L)
        self.inst_norm = nn.LazyInstanceNorm1d(momentum=0.1, affine=True)
        # Pointwise conv to mix channels after normalization and restore a projection
        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=conv3_kernel,
            stride=conv3_stride,
            padding=conv3_padding,
            bias=True,
        )
        # A small projection for the residual path to match channels if needed
        self.res_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

        # activation
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:

        Args:
            x (torch.Tensor): Input tensor of shape (N, in_channels, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (N, out_channels, H_pool, W_pool)
        """
        # Save original for residual projection later
        orig = x

        # Spatial convolutions
        x = self.conv1(x)           # (N, mid_channels, H, W)
        x = self.activation(x)
        x = self.conv2(x)           # (N, out_channels, H, W)
        x = self.activation(x)

        # Reduce spatial resolution
        x = self.avg_pool(x)        # (N, out_channels, H', W')

        N, C, Hp, Wp = x.shape

        # Flatten spatial dims into a sequence length dimension for InstanceNorm1d:
        # InstanceNorm1d normalizes over each channel for each sample across the sequence length.
        seq_len = Hp * Wp
        x_seq = x.view(N, C, seq_len)   # (N, C, L)

        # Apply LazyInstanceNorm1d (will initialize num_features=C on first forward)
        x_seq = self.inst_norm(x_seq)   # (N, C, L)

        # Restore spatial shape
        x = x_seq.view(N, C, Hp, Wp)    # (N, out_channels, H', W')

        # Pointwise conv and activation
        x = self.conv3(x)
        x = self.activation(x)

        # Residual merge: project original input to match channels and downsample spatially to (H', W')
        res = self.res_proj(orig)
        # Downsample residual using average pooling to match pooled dimensions
        res = nn.functional.adaptive_avg_pool2d(res, (Hp, Wp))

        # Merge and final activation
        out = x + res
        out = torch.tanh(out)  # bounded output

        return out


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    """
    Returns the initialization parameters required to construct the Model in order:
      [in_channels, mid_channels, out_channels, conv1_kernel, conv1_stride, conv1_padding,
       conv2_kernel, conv2_stride, conv2_padding, pool_kernel, pool_stride, pool_padding,
       conv3_kernel, conv3_stride, conv3_padding]
    """
    return [
        in_channels,
        mid_channels,
        out_channels,
        conv1_kernel,
        conv1_stride,
        conv1_padding,
        conv2_kernel,
        conv2_stride,
        conv2_padding,
        pool_kernel,
        pool_stride,
        pool_padding,
        conv3_kernel,
        conv3_stride,
        conv3_padding,
    ]