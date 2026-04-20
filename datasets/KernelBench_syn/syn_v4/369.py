import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class Model(nn.Module):
    """
    Complex 3D processing block that demonstrates a sequence of operations:
    - Zero padding (nn.ZeroPad3d)
    - 3D convolution (nn.Conv3d)
    - Batch normalization (nn.BatchNorm3d)
    - Channel-wise squeeze-and-excitation style scaling using adaptive avg pool, a linear layer and Tanh activation (nn.Tanh)
    - Spatial dropout across channels (nn.Dropout3d)
    - Residual connection (optionally uses a 1x1x1 conv to match channels)

    This creates a non-trivial combination of layers and tensor reshaping operations.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        pad: Tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0),
        dropout_prob: float = 0.2
    ):
        """
        Initializes the module.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Convolutional kernel size (assumed cubic).
            stride: Convolution stride.
            pad: ZeroPad3d padding tuple (left, right, top, bottom, front, back).
            dropout_prob: Probability for Dropout3d.
        """
        super(Model, self).__init__()
        # Zero padding layer
        self.pad = nn.ZeroPad3d(pad)

        # Main conv-bn block
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

        # Channel-wise scaling head (SE-like): reduce by factor then expand is common,
        # but to keep parameters simple use a single linear transform across channels.
        self.squeeze_fc = nn.Linear(out_channels, out_channels, bias=True)
        self.act = nn.Tanh()

        # Spatial dropout across channels
        self.dropout = nn.Dropout3d(p=dropout_prob)

        # Residual path: if channel counts differ, use 1x1x1 conv to match
        if in_channels != out_channels or stride != 1:
            self.res_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            self.res_bn = nn.BatchNorm3d(out_channels)
        else:
            self.res_conv = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the block.

        Steps:
        1. Zero pad the input.
        2. Apply conv3d -> bn.
        3. Compute channel descriptors via adaptive avg pool and squeeze to shape (B, C).
        4. Pass descriptors through a linear layer and Tanh to obtain per-channel scaling factors.
        5. Reshape scaling and apply multiplicative gating to convolution output.
        6. Apply Dropout3d.
        7. Add residual connection (with optional 1x1 conv) and return result.

        Args:
            x: Input tensor of shape (B, C_in, D, H, W).

        Returns:
            Tensor of shape (B, C_out, D_out, H_out, W_out).
        """
        # 1) Pad
        x_padded = self.pad(x)

        # 2) Conv + BN
        conv_out = self.conv(x_padded)
        conv_out = self.bn(conv_out)

        # 3) Channel descriptor via global average pooling to (B, C, 1, 1, 1)
        desc = F.adaptive_avg_pool3d(conv_out, output_size=(1, 1, 1)).view(conv_out.size(0), conv_out.size(1))  # (B, C)

        # 4) Linear + Tanh to produce a gating vector in (-1, 1)
        gate = self.squeeze_fc(desc)  # (B, C)
        gate = self.act(gate)         # (B, C)

        # Convert gating to multiplicative scale around 1: scale = 1 + gate * 0.5 (bounded)
        scale = 1.0 + 0.5 * gate      # (B, C)

        # 5) Reshape scale to (B, C, 1, 1, 1) and apply
        scale = scale.view(conv_out.size(0), conv_out.size(1), 1, 1, 1)
        gated = conv_out * scale

        # 6) Dropout3d (drops entire channels)
        dropped = self.dropout(gated)

        # 7) Residual add (with matching conv/bn if necessary)
        if self.res_conv is not None:
            res = self.res_conv(x)
            res = self.res_bn(res)
        else:
            res = x  # same shape expected

        # Ensure shapes match for addition (they should if stride and channels handled)
        out = dropped + res
        return out

# Module-level configuration variables
batch_size = 8
in_channels = 16
out_channels = 32
depth = 10
height = 20
width = 20
kernel_size = 3
stride = 2
pad = (1, 1, 1, 1, 1, 1)  # left, right, top, bottom, front, back
dropout_prob = 0.25

def get_inputs():
    """
    Returns a list containing one input tensor matching the configured shapes.
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for the Model constructor in the expected order.
    """
    return [in_channels, out_channels, kernel_size, stride, pad, dropout_prob]