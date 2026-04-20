import torch
import torch.nn as nn

# Configuration variables
batch_size = 8
in_channels = 3
out_channels = 16
height = 64
width = 64

conv_kernel = 3
conv_stride = 1
conv_padding = 1
conv_dilation = 1

pool_kernel = 2
pool_stride = 2

# Depth to create for 3D instance normalization (must divide the pooled height)
depth = 4

class Model(nn.Module):
    """
    Complex model that combines Conv2d, ReLU, MaxPool2d, reshaping into a 3D volume,
    InstanceNorm3d over the (D, H, W) volume per channel, and reshapes back to 2D feature maps.

    Computation pipeline:
        input (N, C_in, H, W)
        -> Conv2d -> ReLU -> MaxPool2d  (N, C_out, Hp, Wp)
        -> reshape to (N, C_out, D, Hp/D, Wp)
        -> InstanceNorm3d over (C_out, D, Hp/D, Wp)
        -> reshape back to (N, C_out, Hp, Wp)
        -> return normalized 2D feature maps
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        pool_kernel: int,
        pool_stride: int,
        depth: int,
    ):
        """
        Initializes the model components.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels (for Conv2d and InstanceNorm3d).
            kernel_size (int): Kernel size for Conv2d.
            stride (int): Stride for Conv2d.
            padding (int): Padding for Conv2d.
            dilation (int): Dilation for Conv2d.
            pool_kernel (int): Kernel size for MaxPool2d.
            pool_stride (int): Stride for MaxPool2d.
            depth (int): Number of slices to create along a pseudo-depth dimension for InstanceNorm3d.
        """
        super(Model, self).__init__()
        self.depth = depth

        # 2D convolution
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

        # Non-linearity
        self.relu = nn.ReLU(inplace=True)

        # Spatial downsampling
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride)

        # Instance normalization over 3D volume (C, D, H, W) per sample
        # The num_features for InstanceNorm3d is the number of channels after Conv2d
        self.inst_norm = nn.InstanceNorm3d(num_features=out_channels, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid 2D->3D->2D pipeline.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (N, C_out, H_pooled, W_pooled) after InstanceNorm3d.
        """
        # 2D convolution + activation
        x = self.conv(x)
        x = self.relu(x)

        # Spatial pooling
        x = self.pool(x)  # shape: (N, C_out, H_p, W_p)

        N, C, H_p, W_p = x.shape

        # Ensure we can split the pooled height into the requested depth slices
        if H_p % self.depth != 0:
            # To avoid silent shape errors, raise with a helpful message.
            raise ValueError(f"Pooled height ({H_p}) is not divisible by depth ({self.depth}).")

        sliced_H = H_p // self.depth  # new height per depth slice

        # Reshape into a 5D tensor suitable for InstanceNorm3d: (N, C, D, H', W)
        x = x.view(N, C, self.depth, sliced_H, W_p)

        # Apply InstanceNorm3d over (C, D, H', W)
        x = self.inst_norm(x)

        # Reshape back to 4D tensor: (N, C, H_p, W_p)
        x = x.view(N, C, H_p, W_p)

        return x

def get_inputs():
    """
    Returns a list containing a single input tensor for the model.

    The shape matches the module-level configuration variables.
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters for the Model constructor in the required order.
    """
    return [
        in_channels,
        out_channels,
        conv_kernel,
        conv_stride,
        conv_padding,
        conv_dilation,
        pool_kernel,
        pool_stride,
        depth,
    ]