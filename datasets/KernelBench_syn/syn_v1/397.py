import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    1D sequence processing model that demonstrates a small feature extractor pipeline:
      - Zero-padding on the temporal dimension
      - 1D convolution to mix channels and local context
      - Lazy BatchNorm1d (initialized at first forward)
      - Hardshrink non-linearity to sparsify activations
      - Adaptive average pooling to aggregate temporal features
      - Fully connected projection to produce final outputs

    This combines padding, normalization, non-linearity and pooling in a compact pipeline.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int,
        pad: int,
        shrink_lambda: float,
        out_features: int
    ):
        """
        Initializes the Model components.

        Args:
            in_channels (int): Number of input channels.
            hidden_channels (int): Number of channels produced by the convolution.
            kernel_size (int): Size of the 1D convolution kernel.
            pad (int): Amount of zero padding applied on each side of the temporal dimension.
            shrink_lambda (float): Lambda parameter for Hardshrink activation.
            out_features (int): Number of output features produced by the final linear layer.
        """
        super(Model, self).__init__()
        # Pad the temporal dimension (last dimension) with zeros on both sides
        self.pad = nn.ZeroPad1d(pad)

        # 1D convolution to extract local temporal features and mix channels
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=kernel_size, bias=False)

        # Lazy BatchNorm1d: will infer num_features on first forward pass
        self.bn = nn.LazyBatchNorm1d()

        # Hardshrink non-linearity to sparsify activations
        self.shrink = nn.Hardshrink(lambd=shrink_lambda)

        # Aggregate temporal dimension to a single value per channel
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)

        # Final projection from aggregated hidden representation to desired outputs
        self.fc = nn.Linear(hidden_channels, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation.

        Steps:
          1. ZeroPad1d on temporal axis
          2. Conv1d to compute local features
          3. Lazy BatchNorm1d (initialized on first call)
          4. Hardshrink activation
          5. Adaptive average pooling to collapse temporal axis
          6. Linear projection to final outputs

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, seq_len)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features)
        """
        # 1. zero-pad the temporal dimension
        x = self.pad(x)

        # 2. convolution
        x = self.conv(x)

        # 3. batch normalization (lazy)
        x = self.bn(x)

        # 4. sparse activation
        x = self.shrink(x)

        # 5. temporal aggregation -> shape (batch, channels, 1)
        x = self.pool(x)

        # remove the last dimension and project
        x = x.squeeze(-1)  # shape -> (batch, hidden_channels)

        # 6. final fully-connected projection
        out = self.fc(x)
        return out

# Configuration variables
batch_size = 8
in_channels = 12
hidden_channels = 32
seq_len = 128
kernel_size = 5
pad = 2  # pad on both sides so conv can preserve length when kernel_size is odd
shrink_lambda = 0.5
out_features = 10

def get_inputs():
    """
    Returns a list containing a single input tensor matching the
    expected shape (batch_size, in_channels, seq_len).
    """
    x = torch.randn(batch_size, in_channels, seq_len)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters for the Model constructor in order.
    """
    return [in_channels, hidden_channels, kernel_size, pad, shrink_lambda, out_features]