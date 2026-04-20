import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex 1D sequence processing model that combines:
      - Lazy 1D convolution (nn.LazyConv1d) to lazily infer input channels,
      - Group Normalization (nn.GroupNorm) applied on conv outputs,
      - Nonlinear activation (GELU),
      - A multi-layer (optionally bidirectional) LSTM (nn.LSTM) over the time dimension,
      - A linear projection applied per time-step, followed by global average pooling.

    The forward pass expects input of shape (batch_size, in_channels, seq_length) and
    returns a tensor of shape (batch_size, linear_out_features).
    """
    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        gn_groups: int = 8,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        bidirectional: bool = True,
        linear_out: int = 128
    ):
        super(Model, self).__init__()

        # Convolution with lazy in_channels inference; only out_channels is required.
        self.conv = nn.LazyConv1d(
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )

        # GroupNorm expects num_channels == out_channels
        self.gn = nn.GroupNorm(num_groups=gn_groups, num_channels=out_channels)

        # Non-linear activation
        self.act = nn.GELU()

        # LSTM works on (batch, seq, feature) when batch_first=True
        self.lstm = nn.LSTM(
            input_size=out_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        # Linear projection applied to each time-step output of LSTM
        lstm_out_features = lstm_hidden * (2 if bidirectional else 1)
        self.proj = nn.Linear(lstm_out_features, linear_out)

        # Global pooling across the temporal dimension after projection
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, in_channels, seq_length)

        Returns:
            Tensor of shape (batch_size, linear_out_features)
        """
        # 1) Convolution over temporal dimension -> (B, out_channels, T')
        x = self.conv(x)

        # 2) Group normalization (operates on (B, C, L))
        x = self.gn(x)

        # 3) Non-linearity
        x = self.act(x)

        # 4) Prepare for LSTM: (B, T', C)
        x = x.permute(0, 2, 1)

        # 5) LSTM over temporal sequence -> (B, T', hidden * directions)
        lstm_out, _ = self.lstm(x)

        # 6) Linear projection applied per time-step -> (B, T', linear_out)
        proj = self.proj(lstm_out)

        # 7) Pool across time: move to (B, linear_out, T') and adaptive pool to length 1
        proj = proj.permute(0, 2, 1)
        pooled = self.pool(proj).squeeze(-1)  # (B, linear_out)

        return pooled

# Configuration variables
batch_size = 8
in_channels = 3  # lazy conv will infer this, but used to create input tensor
seq_length = 512

out_channels = 32
conv_kernel_size = 5
conv_stride = 2
conv_padding = 2
conv_dilation = 1
conv_groups = 1
conv_bias = True

gn_groups = 8  # must divide out_channels (32)

lstm_hidden = 64
lstm_layers = 2
lstm_bidirectional = True

linear_out_features = 128

def get_inputs():
    """
    Returns runtime input tensors to be passed to Model.forward.
    """
    x = torch.randn(batch_size, in_channels, seq_length)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for the Model constructor in the same order
    as the __init__ signature expects them (except 'self').

    Order:
      out_channels, kernel_size, stride, padding, dilation, groups, bias,
      gn_groups, lstm_hidden, lstm_layers, bidirectional, linear_out
    """
    return [
        out_channels,
        conv_kernel_size,
        conv_stride,
        conv_padding,
        conv_dilation,
        conv_groups,
        conv_bias,
        gn_groups,
        lstm_hidden,
        lstm_layers,
        lstm_bidirectional,
        linear_out_features
    ]