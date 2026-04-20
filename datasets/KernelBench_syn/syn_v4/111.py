import torch
import torch.nn as nn
from typing import Tuple

class Model(nn.Module):
    """
    Complex module that combines AdaptiveMaxPool2d, InstanceNorm1d and GLU with a linear
    projection. The model:
      1. Applies adaptive max pooling to reduce spatial dimensions.
      2. Treats the pooled spatial grid as a sequence and normalizes channels with InstanceNorm1d.
      3. Projects per-step channel features to twice the channel dimensionality.
      4. Uses GLU to gate and reduce back to channel dimensionality.
      5. Aggregates the sequence (mean) to produce a per-batch feature vector.

    Input: Tensor of shape (N, C, H, W)
    Output: Tensor of shape (N, C)
    """
    def __init__(self, in_channels: int, output_size: Tuple[int, int], glu_dim: int = -1):
        """
        Initializes the model components.

        Args:
            in_channels (int): Number of input channels C.
            output_size (tuple): (out_h, out_w) target size for adaptive pooling.
            glu_dim (int, optional): Dimension along which GLU splits. Defaults to -1 (last dim).
        """
        super(Model, self).__init__()
        self.pool = nn.AdaptiveMaxPool2d(output_size=output_size)
        # InstanceNorm1d normalizes over the channel dimension for each instance across the sequence length.
        self.inst_norm = nn.InstanceNorm1d(num_features=in_channels, affine=True)
        # Linear projects per-sequence-step features from C -> 2*C so GLU can split into value & gate.
        self.linear = nn.Linear(in_channels, in_channels * 2, bias=True)
        self.glu = nn.GLU(dim=glu_dim)
        self._out_h, self._out_w = output_size
        self._in_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining pooling, normalization, linear projection and GLU gating.

        Args:
            x (torch.Tensor): Input tensor with shape (N, C, H, W).

        Returns:
            torch.Tensor: Output tensor with shape (N, C) after sequence aggregation.
        """
        # 1) Adaptive spatial pooling -> (N, C, out_h, out_w)
        pooled = self.pool(x)

        # 2) Flatten spatial dims into a sequence length L = out_h * out_w -> (N, C, L)
        N, C, H, W = pooled.shape
        seq = pooled.view(N, C, H * W)

        # 3) Instance normalization across the channel dimension for each instance
        seq = self.inst_norm(seq)  # (N, C, L)

        # 4) Reorder to (N, L, C) to apply a per-step linear projection
        seq = seq.permute(0, 2, 1)  # (N, L, C)
        proj = self.linear(seq)     # (N, L, 2*C)

        # 5) GLU gating reduces last dim from 2*C -> C
        gated = self.glu(proj)      # (N, L, C)

        # 6) Aggregate sequence dimension (mean) to produce a compact representation per batch element
        out = gated.mean(dim=1)     # (N, C)
        return out

# Configuration / default inputs
batch_size = 8
channels = 64  # must be even (GLU will split into two halves)
height = 32
width = 32
out_h = 4
out_w = 4
glu_dim = -1  # GLU will split on the last dimension (after linear projection)

def get_inputs():
    """
    Returns example input tensors for the model.
    """
    x = torch.randn(batch_size, channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for the Model constructor.
    """
    return [channels, (out_h, out_w), glu_dim]