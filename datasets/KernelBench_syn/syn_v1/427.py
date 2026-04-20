import torch
import torch.nn as nn
import torch.nn.functional as F

# Configuration (module-level)
BATCH_SIZE = 8
IN_CHANNELS = 16
OUT_CHANNELS = 32
HEIGHT = 64
WIDTH = 64
PAD = 1               # Reflection padding size (for 3x3 neighborhood)
DROPOUT_P = 0.15      # Dropout2d probability
THRESHOLD = 0.01      # Threshold value for nn.Threshold
THRESHOLD_REPLACEMENT = 0.0  # Value to replace inputs below threshold

class Model(nn.Module):
    """
    Patch-aggregation module that:
      - Applies reflection padding to the input image tensor,
      - Extracts 3x3 patches (sliding window),
      - Linearly mixes each patch across channels to produce new channel embeddings,
      - Applies Channel-wise Dropout (Dropout2d),
      - Applies a Threshold non-linearity,
      - Applies an external channel gate (per-sample, per-channel scaling).

    Forward Args:
        x (torch.Tensor): Input of shape (B, C_in, H, W)
        gate (torch.Tensor): Channel gating tensor of shape (B, C_out) or (B, C_out, 1, 1)

    Returns:
        torch.Tensor: Output tensor of shape (B, C_out, H, W)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pad: int = PAD,
        dropout_p: float = DROPOUT_P,
        threshold: float = THRESHOLD,
        threshold_replacement: float = THRESHOLD_REPLACEMENT,
        bias: bool = True,
    ):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pad = pad
        self.kernel_size = 3  # fixed 3x3 patch aggregation

        # Layers from the allowed set
        self.reflection_pad = nn.ReflectionPad2d(self.pad)
        self.dropout2d = nn.Dropout2d(p=dropout_p)
        self.threshold = nn.Threshold(threshold, threshold_replacement)

        # Learned linear mixing of flattened patches (C_in * kernel_area -> C_out)
        weight_shape = (in_channels * (self.kernel_size ** 2), out_channels)
        self.weight = nn.Parameter(torch.empty(weight_shape))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # Initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, H, W)
        gate: (B, C_out) or (B, C_out, 1, 1) - per-sample per-channel scaling after processing
        """
        B, C_in, H, W = x.shape
        assert C_in == self.in_channels, f"Input channels ({C_in}) must match Model.in_channels ({self.in_channels})"

        # 1) Reflection padding to ensure 3x3 patches are defined at borders
        x_padded = self.reflection_pad(x)  # (B, C_in, H + 2*pad, W + 2*pad)

        # 2) Extract 3x3 patches via F.unfold -> (B, C_in * 9, L) where L = H*W
        patches = F.unfold(x_padded, kernel_size=self.kernel_size, stride=1)  # (B, C_in*9, L)
        L = patches.shape[-1]  # should equal H * W

        # 3) Linear mix: for each spatial location, mix the flattened patch vector into out_channels
        #    Move spatial dim to middle for matmul: (B, L, C_in*9)
        patches = patches.transpose(1, 2)  # (B, L, C_in*9)
        out_flat = patches @ self.weight  # (B, L, C_out)
        if self.bias is not None:
            out_flat = out_flat + self.bias.unsqueeze(0).unsqueeze(0)  # broadcast bias to (B, L, C_out)

        # 4) Reshape back to (B, C_out, H, W)
        out = out_flat.transpose(1, 2).contiguous().view(B, self.out_channels, H, W)

        # 5) Channel-wise dropout (stochastic channel dropping)
        out = self.dropout2d(out)

        # 6) Threshold non-linearity to suppress tiny activations
        out = self.threshold(out)

        # 7) Apply external gating: support (B, C_out) or (B, C_out, 1, 1)
        if gate.dim() == 2:
            gate = gate.unsqueeze(-1).unsqueeze(-1)  # -> (B, C_out, 1, 1)
        assert gate.shape == (B, self.out_channels, 1, 1), f"Gate must broadcast to (B, {self.out_channels}, 1, 1)"
        out = out * gate

        return out

import math

# Test configuration variables (matching module-level ones above)
batch_size = BATCH_SIZE
in_channels = IN_CHANNELS
out_channels = OUT_CHANNELS
height = HEIGHT
width = WIDTH
pad = PAD
dropout_p = DROPOUT_P
threshold = THRESHOLD
threshold_replacement = THRESHOLD_REPLACEMENT

def get_inputs():
    # Input image tensor
    x = torch.randn(batch_size, in_channels, height, width)

    # Create a per-sample, per-channel gate between 0.0 and 1.0
    gate = torch.rand(batch_size, out_channels)

    return [x, gate]

def get_init_inputs():
    # Arguments required to initialize the Model
    return [in_channels, out_channels, pad, dropout_p, threshold, threshold_replacement, True]