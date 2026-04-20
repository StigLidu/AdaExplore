import torch
import torch.nn as nn

# Module-level configuration
batch_size = 8
in_channels = 32
hidden_channels = 48
out_channels = 32
height = 64
width = 64
kernel_size = 3
stride = 1
padding = None  # if None, will default to kernel_size // 2 to preserve spatial dims
use_tanh_gate = True

class Model(nn.Module):
    """
    Complex 2D feature-transform block combining pointwise reduction, depthwise convolution,
    batch normalization, Hardswish activations, and a tanh-based global gating mechanism.
    The pattern creates a residual-style transformation with an input-conditioned gate.

    Computation graph:
      identity = skip(x)
      x_reduced = conv1x1(x) -> bn -> hardswish
      x_dw = depthwise_conv(x_reduced) -> bn -> hardswish
      x_proj = conv1x1(x_dw) -> bn
      gate = tanh(gate_conv(adaptive_avg_pool(x)))
      out = hardswish( x_proj * gate + identity )
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = None,
        use_tanh_gate: bool = True
    ):
        super(Model, self).__init__()
        if padding is None:
            padding = kernel_size // 2

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.stride = stride
        self.use_tanh_gate = use_tanh_gate

        # Pointwise reduction
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.act1 = nn.Hardswish()

        # Depthwise convolution (spatial feature extractor)
        self.conv_dw = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, stride=stride,
                                 padding=padding, groups=hidden_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.act2 = nn.Hardswish()

        # Pointwise projection to desired output channels
        self.conv_proj = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Skip connection (identity or 1x1 conv to match channels/stride)
        if in_channels == out_channels and stride == 1:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)

        # Global gating path: pool -> 1x1 conv -> tanh
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.gate_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.tanh = nn.Tanh()

        # Final activation
        self.final_act = nn.Hardswish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, H_out, W_out).
        """
        # Save skip / identity
        identity = self.skip(x)

        # Reduction -> depthwise -> projection
        x_reduced = self.conv1(x)
        x_reduced = self.bn1(x_reduced)
        x_reduced = self.act1(x_reduced)

        x_dw = self.conv_dw(x_reduced)
        x_dw = self.bn2(x_dw)
        x_dw = self.act2(x_dw)

        x_proj = self.conv_proj(x_dw)
        x_proj = self.bn3(x_proj)

        # Global gating conditioned on original input
        if self.use_tanh_gate:
            pooled = self.global_pool(x)               # (B, C_in, 1, 1)
            gate = self.gate_conv(pooled)              # (B, out_channels, 1, 1)
            gate = self.tanh(gate)                     # values in (-1, 1)
            out = x_proj * gate                        # channel-wise & spatial broadcasted gating
        else:
            out = x_proj

        # Residual addition and final activation
        out = out + identity
        out = self.final_act(out)
        return out


def get_inputs():
    """
    Returns a list with a single input tensor for the model.
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    """
    Returns a list of initialization parameters for the Model constructor.
    Order matches: in_channels, hidden_channels, out_channels, kernel_size, stride, padding, use_tanh_gate
    """
    return [in_channels, hidden_channels, out_channels, kernel_size, stride, padding, use_tanh_gate]