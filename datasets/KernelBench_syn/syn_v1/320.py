import torch
import torch.nn as nn
from typing import Tuple, List

class Model(nn.Module):
    """
    Complex PyTorch module that demonstrates a small feature-processing pipeline:
      1. 2D constant padding on spatial dimensions (H, W)
      2. Unsqueeze to introduce a depth dimension (D=1)
      3. Adaptive 3D average pooling to a target (D_out, H_out, W_out)
      4. Channel-wise gating computed from spatial averages and sigmoid
      5. Layer normalization across (C, D_out, H_out, W_out)

    The model expects an input tensor of shape (B, C, H, W).
    Initialization requires specification of channels, pad size and the 3D pooling output size.
    """
    def __init__(self, channels: int, pad: int, pool_output_size: Tuple[int, int, int]):
        """
        Args:
            channels (int): Number of channels in the input tensor.
            pad (int): Symmetric padding applied to left/right/top/bottom via ConstantPad2d.
            pool_output_size (tuple): Desired output size from AdaptiveAvgPool3d as (D_out, H_out, W_out).
                                      Note: D_out must be <= 1 because the model unsqueezes a single depth.
        """
        super(Model, self).__init__()
        self.channels = channels
        self.pad = pad
        self.pool_output_size = pool_output_size

        # 2D constant pad applied before introducing a depth dimension.
        self.pad2d = nn.ConstantPad2d(self.pad, 0.0)

        # Adaptive average pooling in 3D after unsqueezing a depth dimension (initial D=1)
        self.pool3d = nn.AdaptiveAvgPool3d(self.pool_output_size)

        # LayerNorm will normalize over (C, D_out, H_out, W_out) for each sample in the batch.
        # normalized_shape must match the trailing dimensions of the input to LayerNorm.
        normalized_shape = (self.channels, self.pool_output_size[0], self.pool_output_size[1], self.pool_output_size[2])
        self.layernorm = nn.LayerNorm(normalized_shape, eps=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation:
          - x: (B, C, H, W)
          - pad spatial dims -> (B, C, H+2*pad, W+2*pad)
          - unsqueeze depth -> (B, C, 1, H_pad, W_pad)
          - adaptive pool -> (B, C, D_out, H_out, W_out)
          - compute channel-wise gate = sigmoid(mean over D,H,W) -> (B, C, 1, 1, 1)
          - scale pooled tensor by gate
          - apply LayerNorm over (C, D_out, H_out, W_out)
        Returns:
          Tensor of shape (B, C, D_out, H_out, W_out)
        """
        # 1) Padding (B, C, H_pad, W_pad)
        x_padded = self.pad2d(x)

        # 2) Introduce depth dimension: (B, C, 1, H_pad, W_pad)
        x_5d = x_padded.unsqueeze(2)

        # 3) Adaptive average pooling to target 3D size
        pooled = self.pool3d(x_5d)  # (B, C, D_out, H_out, W_out)

        # 4) Channel-wise gating: compute spatial averages and apply sigmoid
        gate = pooled.mean(dim=(2, 3, 4), keepdim=True)  # (B, C, 1, 1, 1)
        gate = torch.sigmoid(gate)

        # 5) Scale pooled features and apply LayerNorm
        scaled = pooled * gate
        out = self.layernorm(scaled)

        return out

# Configuration for inputs and initialization
batch_size = 8
channels = 32
height = 64
width = 64

# Pad applied to H and W (symmetric on all sides)
pad = 3

# Desired 3D output size after AdaptiveAvgPool3d.
# Note: Depth must be <= initial depth (which is 1 after unsqueeze), so use depth=1.
pool_output_size = (1, 16, 16)

def get_inputs() -> List[torch.Tensor]:
    """
    Generates a single input tensor suitable for the Model:
      - Shape: (batch_size, channels, height, width)
    """
    x = torch.randn(batch_size, channels, height, width)
    return [x]

def get_init_inputs() -> List:
    """
    Returns the initialization inputs for Model.__init__:
      - channels (int)
      - pad (int)
      - pool_output_size (tuple)
    """
    return [channels, pad, pool_output_size]