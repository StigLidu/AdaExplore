import torch
import torch.nn as nn

class Model(nn.Module):
    """
    1D processing pipeline that demonstrates padding, convolution, pooling/unpooling,
    and channel-wise normalization via LogSoftmax. The model:
      - Applies asymmetric zero-padding to the time dimension
      - Projects inputs with a 1D convolution
      - Performs MaxPool1d (saving indices)
      - Applies LogSoftmax across channel dimension
      - Uses MaxUnpool1d to restore pooled resolution
      - Combines the unpooled signal with the convolutional features and crops to original length
      - Reduces across channels to produce a single-channel output per time step
    """
    def __init__(
        self,
        pad_left: int,
        pad_right: int,
        in_channels: int,
        mid_channels: int,
        pool_kernel: int,
        pool_stride: int
    ):
        """
        Initializes the model components.

        Args:
            pad_left (int): Number of zeros to pad on the left of the sequence.
            pad_right (int): Number of zeros to pad on the right of the sequence.
            in_channels (int): Number of input channels.
            mid_channels (int): Number of channels after the initial convolution.
            pool_kernel (int): Kernel size for MaxPool1d / MaxUnpool1d.
            pool_stride (int): Stride for MaxPool1d / MaxUnpool1d.
        """
        super(Model, self).__init__()
        self.pad_left = pad_left
        self.pad_right = pad_right
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.pool_kernel = pool_kernel
        self.pool_stride = pool_stride

        # Layers
        # Asymmetric zero-padding on the temporal dimension (length axis)
        self.pad = nn.ZeroPad1d((self.pad_left, self.pad_right))
        # Small convolution that preserves temporal length (kernel_size=3, padding=1)
        self.conv = nn.Conv1d(in_channels=self.in_channels, out_channels=self.mid_channels, kernel_size=3, padding=1)
        # MaxPool1d producing indices for unpooling
        self.pool = nn.MaxPool1d(kernel_size=self.pool_kernel, stride=self.pool_stride, return_indices=True)
        # MaxUnpool1d to partially invert the pooling operation
        self.unpool = nn.MaxUnpool1d(kernel_size=self.pool_kernel, stride=self.pool_stride)
        # LogSoftmax across the channel dimension to produce normalized log-probabilities per channel
        self.logsoft = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1, length), a single-channel
                          aggregated signal aligned to the original input length.
        """
        # Preserve original temporal length for final cropping
        orig_length = x.size(2)

        # 1) Asymmetric zero-padding
        x_padded = self.pad(x)  # shape: (N, C_in, length_padded)

        # 2) Convolutional projection
        x_conv = self.conv(x_padded)  # shape: (N, mid_channels, length_padded)

        # 3) Max pooling (save indices for unpool)
        x_pooled, indices = self.pool(x_conv)  # shape: (N, mid_channels, length_pooled)

        # 4) Channel-wise log-softmax to produce normalized log-weights per channel
        x_logsoft = self.logsoft(x_pooled)  # shape: (N, mid_channels, length_pooled)

        # 5) Unpool back to the convolutional temporal resolution using saved indices
        # Provide output_size to ensure correct unpooled length (matches x_conv)
        x_unpooled = self.unpool(x_logsoft, indices, output_size=x_conv.size())  # shape: (N, mid_channels, length_padded)

        # 6) Combine unpooled (normalized) representation with original conv features
        # Element-wise modulation: treat log-probabilities as weights (can be negative) that modulate conv activations
        x_combined = x_unpooled * x_conv  # shape: (N, mid_channels, length_padded)

        # 7) Crop the padded boundaries to match original input length
        start = self.pad_left
        end = start + orig_length
        x_cropped = x_combined[:, :, start:end]  # shape: (N, mid_channels, orig_length)

        # 8) Aggregate across channels to produce a single-channel output per time step
        out = torch.sum(x_cropped, dim=1, keepdim=True)  # shape: (N, 1, orig_length)

        return out

# Configuration / default inputs for testing and initialization
batch_size = 8
in_channels = 6
mid_channels = 12
length = 1024
pad_left = 3
pad_right = 5
pool_kernel = 4
pool_stride = 2

def get_inputs():
    """
    Generates a random input tensor matching the expected shape:
    (batch_size, in_channels, length)
    """
    x = torch.randn(batch_size, in_channels, length)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters corresponding to Model.__init__:
    [pad_left, pad_right, in_channels, mid_channels, pool_kernel, pool_stride]
    """
    return [pad_left, pad_right, in_channels, mid_channels, pool_kernel, pool_stride]