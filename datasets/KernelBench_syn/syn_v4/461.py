import torch
import torch.nn as nn
from typing import List, Any

class Model(nn.Module):
    """
    A moderately complex 2D convolutional module that demonstrates the use of lazy initialization
    and max-unpooling to reconstruct spatial dimensions. The model performs the following steps:

    1. Lazy convolution to project input channels to a fixed number of feature maps.
    2. 1x1 reduction to halve the channel dimension and a ReLU + BatchNorm.
    3. Max pooling with indices (returns indices for unpooling).
    4. A small convolution on the pooled representation.
    5. Max unpooling using the stored indices to restore the pre-pooled spatial dimensions.
    6. A lazy 1x1 convolution on the original input to create a residual path.
    7. Channel-wise concatenation of unpooled features and residual, followed by a final conv.

    This combines nn.LazyConv2d and nn.MaxUnpool2d in a coherent forward pass.
    """
    def __init__(self,
                 out_channels: int,
                 conv_kernel: int = 3,
                 pool_kernel: int = 2,
                 pool_stride: int = 2,
                 pool_padding: int = 0):
        """
        Initializes the model components.

        Args:
            out_channels (int): Number of output channels produced by the initial lazy conv.
            conv_kernel (int, optional): Kernel size for the initial lazy convolution. Defaults to 3.
            pool_kernel (int, optional): Kernel size for MaxPool2d and MaxUnpool2d. Defaults to 2.
            pool_stride (int, optional): Stride for pooling/unpooling. Defaults to 2.
            pool_padding (int, optional): Padding for pooling/unpooling. Defaults to 0.
        """
        super(Model, self).__init__()
        # Primary lazy convolution: input channels are inferred on first forward
        self.lazy_conv = nn.LazyConv2d(out_channels=out_channels, kernel_size=conv_kernel, padding=conv_kernel // 2)

        # Reduce channels to half using a 1x1 conv (in_channels == out_channels)
        reduced_channels = max(1, out_channels // 2)
        self.reduce_conv = nn.Conv2d(in_channels=out_channels, out_channels=reduced_channels, kernel_size=1)

        # Normalization and activation
        self.bn = nn.BatchNorm2d(reduced_channels)
        self.relu = nn.ReLU(inplace=True)

        # Pooling that returns indices for unpooling
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride, padding=pool_padding, return_indices=True)
        # Small post-pool conv to simulate processing in the latent space
        self.post_conv = nn.Conv2d(in_channels=reduced_channels, out_channels=reduced_channels, kernel_size=3, padding=1)

        # Unpool layer that uses the indices from self.pool
        self.unpool = nn.MaxUnpool2d(kernel_size=pool_kernel, stride=pool_stride, padding=pool_padding)

        # Residual mapper from original input channels to reduced_channels (lazy to infer in_channels)
        self.res_conv = nn.LazyConv2d(out_channels=reduced_channels, kernel_size=1)

        # Final convolution to mix concatenated channels back to out_channels
        # concatenated channels = reduced_channels (unpooled) + reduced_channels (residual) = 2 * reduced_channels
        self.final_conv = nn.Conv2d(in_channels=2 * reduced_channels, out_channels=out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        # 1) Project input to a feature space (lazy initialization will set in_channels on first call)
        feat = self.lazy_conv(x)                # -> (B, out_channels, H, W)
        feat = self.relu(feat)

        # 2) Reduce channels and normalize
        reduced = self.reduce_conv(feat)        # -> (B, reduced_channels, H, W)
        reduced = self.bn(reduced)
        reduced = self.relu(reduced)

        # Save shape before pooling for unpool output_size
        pre_pool_shape = reduced.size()         # (B, reduced_channels, H, W)

        # 3) Pool and keep indices for unpooling
        pooled, indices = self.pool(reduced)    # pooled -> (B, reduced_channels, Hp, Wp)

        # 4) Small processing in pooled domain
        processed = self.post_conv(pooled)      # -> (B, reduced_channels, Hp, Wp)
        processed = self.relu(processed)

        # 5) Unpool to restore spatial resolution (use pre_pool_shape as output_size)
        unpooled = self.unpool(processed, indices, output_size=pre_pool_shape)  # -> (B, reduced_channels, H, W)

        # 6) Residual path: map original input to reduced_channels
        residual = self.res_conv(x)             # -> (B, reduced_channels, H, W)
        residual = self.relu(residual)

        # 7) Concatenate unpooled features and residual along channel dimension and finalize
        combined = torch.cat([unpooled, residual], dim=1)  # -> (B, 2 * reduced_channels, H, W)
        out = self.final_conv(combined)                    # -> (B, out_channels, H, W)
        out = self.relu(out)
        return out

# Configuration for inputs and initialization
batch_size = 8
in_channels = 3
height = 32
width = 32

out_channels = 32
conv_kernel = 3
pool_kernel = 2
pool_stride = 2
pool_padding = 0

def get_inputs() -> List[torch.Tensor]:
    """
    Returns a list containing a single input tensor for the model.
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs() -> List[Any]:
    """
    Returns a list of initialization arguments to construct the Model.
    Order: out_channels, conv_kernel, pool_kernel, pool_stride, pool_padding
    """
    return [out_channels, conv_kernel, pool_kernel, pool_stride, pool_padding]