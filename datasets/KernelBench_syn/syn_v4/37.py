import torch
import torch.nn as nn
from typing import Tuple, List, Any

class Model(nn.Module):
    """
    Complex 3D feature fusion model that demonstrates:
    - Adaptive 3D max pooling to a fixed spatial grid
    - Splitting channels into two views and aggregating differently (mean vs max)
    - A bilinear interaction between the two aggregated views
    - Non-linear modulation via Tanhshrink and a learned gating mechanism
    - A final projection to a desired output dimensionality
    """
    def __init__(self, in_channels: int, pool_output_size: Tuple[int, int, int],
                 bilinear_out: int, final_out: int):
        """
        Args:
            in_channels (int): Number of input channels (must be even).
            pool_output_size (tuple): Target (D, H, W) for AdaptiveMaxPool3d.
            bilinear_out (int): Output feature size of the bilinear layer.
            final_out (int): Final output feature dimensionality.
        """
        super(Model, self).__init__()
        assert in_channels % 2 == 0, "in_channels must be even to split into two views."
        self.in_channels = in_channels
        self.half = in_channels // 2
        self.pool_output_size = pool_output_size
        self.bilinear_out = bilinear_out
        self.final_out = final_out

        # Adaptive max pooling to compress spatial dimensions to a fixed grid
        self.pool = nn.AdaptiveMaxPool3d(self.pool_output_size)

        # Bilinear interaction between the two aggregated channel-halves
        # in1_features == in2_features == half
        self.bilinear = nn.Bilinear(self.half, self.half, self.bilinear_out, bias=True)

        # Nonlinearity: Tanhshrink applied after bilinear interaction
        self.tanhshrink = nn.Tanhshrink()

        # A gating mechanism to modulate the bilinear features before final projection
        self.gate = nn.Linear(self.bilinear_out, self.bilinear_out)
        # Final projection to desired output dimensionality
        self.final_proj = nn.Linear(self.bilinear_out, self.final_out)

        # Small normalization for stability
        self.norm = nn.LayerNorm(self.bilinear_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        1. Adaptive max pool -> (B, C, pD, pH, pW)
        2. Split channels into two halves: A and B
        3. Aggregate A by spatial mean, B by spatial max -> (B, half), (B, half)
        4. Bilinear interaction: y = Bilinear(A_agg, B_agg) -> (B, bilinear_out)
        5. Normalize, apply tanhshrink, compute gate and modulate features
        6. Final linear projection to final_out

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, D, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (B, final_out)
        """
        # 1) Pool to fixed spatial grid
        pooled = self.pool(x)  # shape: (B, C, pD, pH, pW)

        # 2) Split channel dimension into two equal views
        a = pooled[:, :self.half, ...]  # (B, half, pD, pH, pW)
        b = pooled[:, self.half:, ...]  # (B, half, pD, pH, pW)

        # 3) Different spatial aggregations:
        #    - a: mean over spatial grid
        #    - b: max over spatial grid
        a_agg = a.mean(dim=(2, 3, 4))   # (B, half)
        b_agg, _ = b.max(dim=4)        # max over W -> (B, half, pD, pH)
        b_agg, _ = b_agg.max(dim=3)    # max over H -> (B, half, pD)
        b_agg, _ = b_agg.max(dim=2)    # max over D -> (B, half)

        # 4) Bilinear interaction between the two aggregated views
        bil = self.bilinear(a_agg, b_agg)  # (B, bilinear_out)

        # 5) Normalize and non-linearity, then compute gate to modulate features
        bil_norm = self.norm(bil)
        bil_nl = self.tanhshrink(bil_norm)           # element-wise nonlinearity
        gate = torch.sigmoid(self.gate(bil_nl))      # gating values in (0,1)
        modulated = bil_nl * gate                    # gated features (B, bilinear_out)

        # 6) Final projection
        out = self.final_proj(modulated)             # (B, final_out)
        return out

# Configuration for inputs / initialization
batch_size = 8
in_channels = 20   # must be even
depth = 12
height = 16
width = 16

pool_d = 4
pool_h = 4
pool_w = 4

bilinear_out = 128
final_out = 256

def get_inputs() -> List[torch.Tensor]:
    """
    Returns a list containing the primary input tensor.
    Shape: (batch_size, in_channels, depth, height, width)
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs() -> List[Any]:
    """
    Returns initialization arguments for the Model constructor in the same order.
    [in_channels, pool_output_size, bilinear_out, final_out]
    """
    return [in_channels, (pool_d, pool_h, pool_w), bilinear_out, final_out]