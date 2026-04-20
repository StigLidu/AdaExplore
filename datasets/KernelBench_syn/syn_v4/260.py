import torch
import torch.nn as nn

# Module-level configuration (can be modified as needed)
batch_size = 8
in_channels = 3
height = 128
width = 128

# PixelUnshuffle downscale factor (spatial reduction)
downscale_factor = 2

# AvgPool1d parameters (operates on the flattened spatial sequence)
pool_kernel = 8
pool_stride = 8
pool_padding = 0

# Small epsilon for numerical stability in RMS normalization
eps = 1e-6


class Model(nn.Module):
    """
    Complex module that demonstrates a mixed pipeline:
      1. PixelUnshuffle to reduce spatial resolution while increasing channels.
      2. Non-linear activation (Hardswish).
      3. Reshape to treat spatial locations as a 1D sequence per channel.
      4. AvgPool1d to aggregate local neighborhoods along the spatial sequence.
      5. RMS normalization across the spatial dimension for each (batch, channel) slice.

    Input:
        x: Tensor of shape (batch_size, in_channels, height, width)

    Output:
        Tensor of shape (batch_size, out_channels, pooled_length)
          where out_channels = in_channels * (downscale_factor ** 2)
                pooled_length = floor((height/downscale) * (width/downscale) pooled by pool params)
    """
    def __init__(self, downscale_factor: int, pool_kernel: int, pool_stride: int = None, pool_padding: int = 0, eps: float = 1e-6):
        """
        Initializes the Model.

        Args:
            downscale_factor (int): Factor to downscale spatial dims with PixelUnshuffle.
            pool_kernel (int): Kernel size for AvgPool1d along the flattened spatial sequence.
            pool_stride (int, optional): Stride for AvgPool1d. If None, stride is set equal to pool_kernel.
            pool_padding (int, optional): Padding for AvgPool1d. Defaults to 0.
            eps (float, optional): Small value for numerical stability in RMS normalization.
        """
        super(Model, self).__init__()
        self.downscale = downscale_factor
        self.eps = eps

        # Layers
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor)
        self.act = nn.Hardswish()

        if pool_stride is None:
            pool_stride = pool_kernel
        # AvgPool1d will operate on (batch, channels_after_unshuffle, sequence_length)
        self.avgpool1d = nn.AvgPool1d(kernel_size=pool_kernel, stride=pool_stride, padding=pool_padding, count_include_pad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass performing the sequence of operations described above.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Tensor after pixel unshuffle, activation, 1D pooling, and RMS normalization.
                          Shape: (N, C * downscale^2, L_out)
        """
        # 1) PixelUnshuffle reduces spatial dimensions by factor r and multiplies channels by r^2
        #    Input: (N, C, H, W) -> Output: (N, C*r^2, H/r, W/r)
        x = self.pixel_unshuffle(x)

        # 2) Non-linear activation applied element-wise
        x = self.act(x)

        # 3) Flatten spatial dimensions into a sequence axis for 1D pooling:
        #    (N, C2, H2, W2) -> (N, C2, L) where L = H2 * W2
        N, C2, H2, W2 = x.shape
        x = x.view(N, C2, H2 * W2)  # contiguous reshape

        # 4) Apply AvgPool1d along the sequence dimension to aggregate local neighborhoods
        x = self.avgpool1d(x)

        # 5) RMS normalization across the spatial (sequence) dimension for each (batch, channel)
        #    rms shape: (N, C2, 1)
        rms = torch.sqrt(torch.mean(x * x, dim=2, keepdim=True) + self.eps)
        x = x / rms

        return x


def get_inputs():
    """
    Generates a representative input tensor for the model.
    Returns:
        List containing a single tensor of shape (batch_size, in_channels, height, width)
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    """
    Returns the initialization arguments for the Model constructor in the same order.
    """
    return [downscale_factor, pool_kernel, pool_stride, pool_padding, eps]