import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex module that combines ZeroPad2d, Upsample and ReLU with a learnable 1x1 mixing
    and channel-wise gating derived from spatial global average pooling.

    Computation pipeline:
      1. Zero-pad the input spatially
      2. Upsample the padded tensor (bilinear)
      3. Apply a learnable 1x1 convolution (via conv2d with weight/bias parameters)
      4. Apply ReLU non-linearity
      5. Compute channel-wise gates from global spatial average and apply sigmoid
      6. Fuse gated activations with a residual-style scaled tanh of the pre-activation
    """
    def __init__(self, in_channels: int, out_channels: int, up_scale: int = 2, pad: tuple = (1, 1, 1, 1)):
        """
        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels produced by the 1x1 mixing convolution.
            up_scale (int): Integer scale factor for spatial upsampling.
            pad (tuple): ZeroPad2d padding as (left, right, top, bottom).
        """
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Zero padding layer
        self.pad = nn.ZeroPad2d(pad)
        # Upsample layer (bilinear for 2D)
        self.up = nn.Upsample(scale_factor=up_scale, mode='bilinear', align_corners=False)
        # ReLU activation
        self.relu = nn.ReLU(inplace=True)

        # Learnable 1x1 convolution parameters (implemented via F.conv2d)
        # Weight shape: (out_channels, in_channels, 1, 1)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, 1, 1) * (1.0 / (in_channels ** 0.5)))
        self.bias = nn.Parameter(torch.zeros(out_channels))

        # Channel gating parameters: scale and bias per output channel (broadcastable over H,W)
        self.gate_scale = nn.Parameter(torch.randn(out_channels, 1, 1) * 0.1)
        self.gate_bias = nn.Parameter(torch.zeros(out_channels, 1, 1))

        # Small scalar for residual-style mixing
        self.res_scale = 0.2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the composed module.

        Args:
            x (torch.Tensor): Input tensor of shape (B, in_channels, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (B, out_channels, H_up, W_up)
        """
        # 1) Zero-pad input
        x_padded = self.pad(x)  # (B, C_in, H_pad, W_pad)

        # 2) Upsample spatially
        x_up = self.up(x_padded)  # (B, C_in, H_up, W_up)

        # 3) 1x1 mixing convolution (learnable)
        x_mix = F.conv2d(x_up, self.weight, bias=self.bias)  # (B, C_out, H_up, W_up)

        # 4) Non-linearity
        x_act = self.relu(x_mix)  # (B, C_out, H_up, W_up)

        # 5) Channel-wise gating: global spatial average -> scale/bias -> sigmoid
        pooled = x_act.mean(dim=[2, 3], keepdim=True)  # (B, C_out, 1, 1)
        gate = torch.sigmoid(self.gate_scale * pooled + self.gate_bias)  # (B, C_out, 1, 1)

        # 6) Fuse gated activations with a small residual from pre-activation (tanh scaled)
        residual = torch.tanh(x_mix) * self.res_scale  # (B, C_out, H_up, W_up)
        out = x_act * gate + residual  # (B, C_out, H_up, W_up)

        return out

# Module-level configuration / example sizes
batch_size = 4
in_channels = 3
out_channels = 16
height = 64
width = 48
up_scale = 2
pad = (1, 2, 1, 2)  # (left, right, top, bottom)

def get_inputs():
    """
    Returns a list containing a single input tensor with the configured shape.
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters for the Model constructor:
    [in_channels, out_channels, up_scale, pad]
    """
    return [in_channels, out_channels, up_scale, pad]