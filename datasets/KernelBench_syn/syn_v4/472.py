import torch
import torch.nn as nn
from typing import Tuple

class Model(nn.Module):
    """
    A composite model that demonstrates a non-trivial dataflow combining:
      - 2D constant padding applied per-slice (using nn.ConstantPad2d)
      - 3D adaptive average pooling (nn.AdaptiveAvgPool3d)
      - A final linear projection with GELU activation

    The model accepts a 5D tensor of shape (batch, channels, depth, height, width).
    To apply ConstantPad2d (which expects a 4D tensor), the model reshapes the input
    to merge batch and depth dimensions, pads the spatial HxW planes, restores the
    original 5D layout, then performs AdaptiveAvgPool3d and a final Linear layer.
    """
    def __init__(
        self,
        in_channels: int,
        pad: Tuple[int, int, int, int],
        output_size: Tuple[int, int, int],
        linear_out_features: int,
        pad_value: float = 0.0
    ):
        """
        Args:
            in_channels (int): Number of channels in the input tensor.
            pad (tuple): 4-tuple (left, right, top, bottom) for ConstantPad2d.
            output_size (tuple): Output size (out_D, out_H, out_W) for AdaptiveAvgPool3d.
            linear_out_features (int): Number of output features for the final Linear layer.
            pad_value (float): Constant pad value for ConstantPad2d. Defaults to 0.0.
        """
        super(Model, self).__init__()
        # ConstantPad2d pads 2D spatial dims (H,W). We'll apply it to each depth-slice.
        self.pad2d = nn.ConstantPad2d(pad, pad_value)
        # AdaptiveAvgPool3d reduces (D,H,W) to the desired output_size
        self.pool3d = nn.AdaptiveAvgPool3d(output_size)
        # Final linear maps flattened pooled features to desired output dimension
        out_d, out_h, out_w = output_size
        linear_in_features = in_channels * out_d * out_h * out_w
        self.fc = nn.Linear(linear_in_features, linear_out_features)
        # small optional activation
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
          1. x: (B, C, D, H, W)
          2. Permute and reshape to (B*D, C, H, W) to apply ConstantPad2d
          3. Restore to (B, C, D, H_pad, W_pad)
          4. Apply AdaptiveAvgPool3d -> (B, C, out_D, out_H, out_W)
          5. Flatten and apply Linear + GELU

        Args:
            x (torch.Tensor): Input tensor with shape (batch, channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor with shape (batch, linear_out_features).
        """
        if x.dim() != 5:
            raise ValueError("Input tensor must be 5D (B, C, D, H, W)")

        B, C, D, H, W = x.shape

        # Merge batch and depth to treat each depth slice as an independent image
        x_slices = x.permute(0, 2, 1, 3, 4).contiguous().view(B * D, C, H, W)

        # Apply 2D constant padding to each (C, H, W) slice
        x_padded = self.pad2d(x_slices)  # (B*D, C, H_pad, W_pad)

        # Recover the 5D shape (B, C, D, H_pad, W_pad)
        H_pad = x_padded.shape[2]
        W_pad = x_padded.shape[3]
        x_restored = x_padded.view(B, D, C, H_pad, W_pad).permute(0, 2, 1, 3, 4).contiguous()

        # Adaptive average pool in 3D to fixed (out_D, out_H, out_W)
        pooled = self.pool3d(x_restored)  # (B, C, out_D, out_H, out_W)

        # Flatten spatial and channel dims
        flattened = pooled.view(B, -1)  # (B, C * out_D * out_H * out_W)

        # Linear projection + GELU
        out = self.fc(flattened)
        out = self.activation(out)
        return out

# Configuration / example sizes
batch_size = 8
in_channels = 16
depth = 10
height = 64
width = 64

# Padding: (left, right, top, bottom)
pad = (1, 2, 3, 4)

# Desired output after AdaptiveAvgPool3d: (out_D, out_H, out_W)
output_size = (3, 8, 8)

# Final linear output dimension
linear_out_features = 128

# Pad value
pad_value = 0.0

def get_inputs():
    """
    Returns example input tensors for running the model.
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Returns the arguments required to initialize the Model.
    Order matches Model.__init__ signature:
      (in_channels, pad, output_size, linear_out_features, pad_value)
    """
    return [in_channels, pad, output_size, linear_out_features, pad_value]