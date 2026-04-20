import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    A composite 1D feature extractor that demonstrates a mix of padding, convolution,
    lazy instance normalization (2D), and parametric ReLU activation. The model:
      1. Applies circular padding to preserve temporal continuity.
      2. Runs a 1D convolution to expand feature channels.
      3. Reshapes the conv output to 4D and applies LazyInstanceNorm2d (lazy initialization).
      4. Applies a channel-wise PReLU.
      5. Performs global average pooling over temporal dimension and a final linear projection.

    Input shape: (batch_size, in_channels, seq_len)
    Output shape: (batch_size, out_dim)
    """
    def __init__(self,
                 in_channels: int,
                 conv_out_channels: int,
                 kernel_size: int,
                 out_dim: int):
        super(Model, self).__init__()

        # Save configuration for possible use in forward or external inspection
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.kernel_size = kernel_size
        self.out_dim = out_dim

        # Circular padding to wrap the temporal boundary
        # Use symmetric padding of kernel_size//2 on both sides to maintain length
        pad = kernel_size // 2
        self.pad = nn.CircularPad1d(pad)

        # 1D convolution after padding. No built-in padding here since we handle it with CircularPad1d.
        self.conv = nn.Conv1d(in_channels, conv_out_channels, kernel_size, bias=True)

        # Lazy instance norm is defined for 2D inputs (N, C, H, W). We will adapt conv output to (N, C, L, 1).
        # LazyInstanceNorm2d will infer num_features on the first forward pass.
        self.inst_norm = nn.LazyInstanceNorm2d()  # num_features set lazily

        # Parametric ReLU with one parameter per output channel (channel-wise)
        self.prelu = nn.PReLU(num_parameters=conv_out_channels)

        # Final projection from pooled channels to desired output dimension
        self.fc = nn.Linear(conv_out_channels, out_dim, bias=True)

        # Initialize conv weights with a scaled normal and bias to zero for stable start
        nn.init.normal_(self.conv.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.conv.bias)
        nn.init.kaiming_uniform_(self.fc.weight, a=math.sqrt(5)) if hasattr(nn.init, 'kaiming_uniform_') else None
        if self.fc.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.fc.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the composite module.

        Args:
            x: Input tensor of shape (batch_size, in_channels, seq_len)

        Returns:
            out: Output tensor of shape (batch_size, out_dim)
        """
        # 1) Circular pad the temporal dimension (works with (N, C, L))
        x_padded = self.pad(x)  # still (N, C, L_padded)

        # 2) 1D convolution to mix channels and local temporal context
        x_conv = self.conv(x_padded)  # shape: (N, conv_out_channels, L)

        # 3) Convert to 4D so that LazyInstanceNorm2d can be applied (N, C, H, W)
        # Treat temporal length as H and add a singleton W dimension.
        x_4d = x_conv.unsqueeze(-1)  # (N, conv_out_channels, L, 1)

        # 4) Instance normalization (lazy initializes num_features on first call)
        x_norm = self.inst_norm(x_4d)  # same shape (N, conv_out_channels, L, 1)

        # 5) Remove the singleton W dimension and apply PReLU activation
        x_norm = x_norm.squeeze(-1)  # (N, conv_out_channels, L)
        x_act = self.prelu(x_norm)   # (N, conv_out_channels, L)

        # 6) Global average pooling over temporal dimension to get a single vector per sample
        x_pool = x_act.mean(dim=2)  # (N, conv_out_channels)

        # 7) Final linear projection
        out = self.fc(x_pool)  # (N, out_dim)

        return out

import math

# Configuration variables (module-level)
batch_size = 8
in_channels = 16
seq_len = 128
conv_out_channels = 32
kernel_size = 5
out_dim = 10

def get_inputs():
    """
    Returns a list of input tensors for the Model.forward.
    Input shape: (batch_size, in_channels, seq_len)
    """
    x = torch.randn(batch_size, in_channels, seq_len, dtype=torch.float32)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for constructing the model externally if needed.
    In this design, the model __init__ requires:
        (in_channels, conv_out_channels, kernel_size, out_dim)
    """
    return [in_channels, conv_out_channels, kernel_size, out_dim]