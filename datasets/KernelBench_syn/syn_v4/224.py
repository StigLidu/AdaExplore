import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    1D sequence processor that demonstrates a small processing pipeline combining:
      - Circular padding (nn.CircularPad1d)
      - 1D convolution (nn.Conv1d)
      - SELU activation (nn.SELU)
      - Average pooling (nn.AvgPool1d)
      - Tanh activation (nn.Tanh)
      - Global aggregation and two small linear heads used for gating and projection

    The forward pass:
      x_pad     = CircularPad1d(x)
      conv_out  = Conv1d(x_pad)
      act       = SELU(conv_out)
      pooled    = AvgPool1d(act)
      nonlin    = Tanh(pooled)
      global_v  = mean(nonlin, dim=2)                # collapse temporal dimension
      proj      = fc(global_v)                       # projection output
      gate      = sigmoid(gate_fc(global_v))         # gating vector
      out       = proj * gate                        # gated projection (element-wise)
    """
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_features: int,
        kernel_size: int = 5,
        pad_size: int = 2,
        pool_kernel: int = 4,
        pool_stride: int = 2,
    ):
        """
        Initializes the model components.

        Args:
            in_channels: Number of input channels.
            mid_channels: Number of channels after the convolution.
            out_features: Size of the final output vector per batch element.
            kernel_size: Convolution kernel size (default 5).
            pad_size: Circular padding applied to both sides before convolution (default 2).
            pool_kernel: Kernel size for average pooling (default 4).
            pool_stride: Stride for average pooling (default 2).
        """
        super(Model, self).__init__()
        # Padding layer: circular pad on the temporal axis
        self.pad = nn.CircularPad1d(pad_size)

        # Convolution: kernel applied after circular padding; padding in conv is 0 since pad handled above
        self.conv = nn.Conv1d(in_channels, mid_channels, kernel_size=kernel_size, stride=1, padding=0, bias=True)

        # Non-linearities
        self.selu = nn.SELU()
        self.tanh = nn.Tanh()

        # Pooling to reduce temporal dimension
        self.avgpool = nn.AvgPool1d(kernel_size=pool_kernel, stride=pool_stride)

        # Two small linear heads: one for projection, one to produce gating values
        self.fc = nn.Linear(mid_channels, out_features, bias=True)
        self.gate_fc = nn.Linear(mid_channels, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, in_channels, seq_length)

        Returns:
            Tensor of shape (batch_size, out_features)
        """
        # 1) Circular padding to preserve border information for convolution
        x_pad = self.pad(x)

        # 2) Convolution + SELU
        conv_out = self.conv(x_pad)
        activated = self.selu(conv_out)

        # 3) Average pooling to reduce temporal resolution
        pooled = self.avgpool(activated)

        # 4) A final non-linearity
        nonlin = self.tanh(pooled)

        # 5) Global temporal aggregation (mean over time dimension)
        #    Result shape: (batch_size, mid_channels)
        global_feat = torch.mean(nonlin, dim=2)

        # 6) Projection and gating
        proj = self.fc(global_feat)                     # (batch, out_features)
        gate = torch.sigmoid(self.gate_fc(global_feat)) # gating in (0,1)

        # 7) Gated output
        out = proj * gate

        return out

# Configuration / default shapes
batch_size = 32
in_channels = 16
seq_length = 1024

mid_channels = 64
out_features = 128
kernel_size = 5
pad_size = 2
pool_kernel = 4
pool_stride = 2

def get_inputs():
    # Random input tensor simulating a batch of 1D sequences
    x = torch.randn(batch_size, in_channels, seq_length, dtype=torch.float32)
    return [x]

def get_init_inputs():
    # Parameters that would be passed when constructing the Model
    return [in_channels, mid_channels, out_features, kernel_size, pad_size, pool_kernel, pool_stride]