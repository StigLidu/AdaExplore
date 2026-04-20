import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex module that:
    - Uses a lazy ConvTranspose2d to expand spatial resolution and set output channels lazily.
    - Applies Hardswish non-linearity.
    - Applies MaxPool2d to reduce spatial resolution.
    - Projects the original input into the same channel space with a 1x1 conv and adds a residual connection.
    - Applies a learnable per-channel scale and returns a global-pooled feature vector.

    Forward pattern:
      x -> LazyConvTranspose2d (upsample & change channels) -> Hardswish -> MaxPool2d -> scale
      original x -> interpolate to match pooled spatial dims -> 1x1 conv -> residual add
      -> global average pool -> output vector (batch_size, out_channels)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride_up: int = 2,
        pool_kernel: int = 2,
    ):
        """
        Initializes the model.

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Desired number of output channels from the deconvolution.
            kernel_size (int): Kernel size for the transpose convolution.
            stride_up (int): Stride for the transpose convolution (controls upsampling).
            pool_kernel (int): Kernel size (and stride) for max pooling.
        """
        super(Model, self).__init__()

        # LazyConvTranspose2d will infer in_channels on first forward; out_channels is known.
        padding = kernel_size // 2
        output_padding = max(0, stride_up - 1)
        self.deconv = nn.LazyConvTranspose2d(
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride_up,
            padding=padding,
            output_padding=output_padding,
        )

        # 1x1 projection for the residual path (requires known in_channels)
        self.skip_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Non-linearity and pooling from the provided list
        self.act = nn.Hardswish()
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_kernel)

        # Learnable per-channel scaling (out_channels is known)
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_channels, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels) after global pooling.
        """
        # Upsample and change channel dimension via lazy transpose convolution
        y = self.deconv(x)  # -> (B, out_channels, H_up, W_up)

        # Non-linearity
        y = self.act(y)

        # Spatial reduction
        y = self.pool(y)  # -> (B, out_channels, H_pooled, W_pooled)

        # Prepare residual by resizing original input to match y's spatial dims
        res_spatial = y.shape[2:]
        res = F.interpolate(x, size=res_spatial, mode='bilinear', align_corners=False)
        res = self.skip_proj(res)  # -> (B, out_channels, H_pooled, W_pooled)

        # Channel-wise learned scaling and residual addition
        y = y * self.scale + res

        # Global average pooling to produce a compact feature vector
        out = y.mean(dim=(2, 3))  # -> (B, out_channels)

        # Final non-linearity for output vector
        out = self.act(out)

        return out

# Configuration / default inputs
batch_size = 8
in_channels = 3
in_h = 32
in_w = 32

out_channels = 16
kernel_size = 3
stride_up = 2
pool_kernel = 2

def get_inputs():
    """
    Returns a list with a single input tensor shaped according to the module-level configuration.
    """
    x = torch.randn(batch_size, in_channels, in_h, in_w)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters for Model.__init__ in the correct order:
    [in_channels, out_channels, kernel_size, stride_up, pool_kernel]
    """
    return [in_channels, out_channels, kernel_size, stride_up, pool_kernel]