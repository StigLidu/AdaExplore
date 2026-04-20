import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

class Model(nn.Module):
    """
    Complex 3D feature aggregator that:
      - Applies an adaptive 3D average pooling to reduce spatial resolution.
      - Performs a learned channel-space projection on the pooled spatial positions
        (per-position channel mixing via matrix multiplication).
      - Applies a GELU non-linearity, aggregates positional features (mean), then
        processes the aggregate with a SELU non-linearity.
      - Projects back to the pooled channel-space, forms a gated reconstruction,
        and upsamples back to the original input resolution.

    This pattern mixes spatial pooling, per-position channel projection, two activations
    (GELU + SELU), and gating to create a non-trivial computation graph.
    """
    def __init__(self, in_channels: int, hidden_dim: int, pool_output: Tuple[int, int, int]):
        """
        Args:
            in_channels: Number of input channels.
            hidden_dim: Hidden dimensionality for aggregated representation.
            pool_output: Tuple (pd, ph, pw) target size for AdaptiveAvgPool3d.
        """
        super(Model, self).__init__()
        if not (isinstance(pool_output, (tuple, list)) and len(pool_output) == 3):
            raise ValueError("pool_output must be a tuple/list of three integers (pd, ph, pw)")

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.pool_output = tuple(int(x) for x in pool_output)

        # Layers
        self.pool = nn.AdaptiveAvgPool3d(self.pool_output)   # reduces spatial dims to pool_output
        self.gelu = nn.GELU()                                # non-linear activation after per-pos proj
        self.selu = nn.SELU()                                # non-linear activation for aggregated vector

        # Learned projections:
        # proj1 mixes channels at each pooled spatial position: (C) -> (hidden_dim)
        pd, ph, pw = self.pool_output
        self.pos_count = pd * ph * pw
        # Using Parameters to emphasize custom linear mappings (equivalent of small dense layers)
        self.proj1 = nn.Parameter(torch.randn(self.in_channels, self.hidden_dim) * (1.0 / max(1, self.in_channels)**0.5))
        self.proj2 = nn.Parameter(torch.randn(self.hidden_dim, self.in_channels * self.pos_count) * (1.0 / max(1, self.hidden_dim)**0.5))

        # Small bias terms for stability
        self.bias1 = nn.Parameter(torch.zeros(self.hidden_dim))
        self.bias2 = nn.Parameter(torch.zeros(self.in_channels * self.pos_count))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, D, H, W)

        Returns:
            Tensor upsampled back to (B, C, D, H, W) with gated reconstruction.
        """
        if x.dim() != 5:
            raise ValueError("Input tensor must be 5D (B, C, D, H, W)")

        B, C, D, H, W = x.shape
        if C != self.in_channels:
            raise ValueError(f"Expected input with {self.in_channels} channels, got {C}")

        # 1) Adaptive average pooling to compact spatial information
        pooled = self.pool(x)  # shape: (B, C, pd, ph, pw)
        pd, ph, pw = self.pool_output
        S = self.pos_count  # pd * ph * pw

        # 2) Rearrange to (B, S, C) for per-position channel mixing
        pooled_flat = pooled.view(B, C, S).permute(0, 2, 1)  # (B, S, C)

        # 3) Per-position channel projection: (B, S, C) @ (C, hidden) -> (B, S, hidden)
        pos_mixed = torch.matmul(pooled_flat, self.proj1) + self.bias1  # (B, S, hidden)
        pos_activated = self.gelu(pos_mixed)  # GELU activation

        # 4) Aggregate positional features into a single vector per-batch
        aggregated = pos_activated.mean(dim=1)  # (B, hidden)

        # 5) Non-linearity on aggregated vector
        aggregated = self.selu(aggregated)  # (B, hidden)

        # 6) Project aggregated vector back to pooled channel-space: (B, hidden) @ (hidden, C*S) -> (B, C*S)
        recon = torch.matmul(aggregated, self.proj2) + self.bias2
        recon = recon.view(B, C, pd, ph, pw)  # (B, C, pd, ph, pw)

        # 7) Form a gating mask from the reconstruction and apply to pooled features
        gate = torch.sigmoid(recon)  # values in (0,1)
        gated = pooled * gate + 0.1 * recon  # residual-ish combination

        # 8) Upsample back to input spatial resolution using trilinear interpolation
        out = F.interpolate(gated, size=(D, H, W), mode='trilinear', align_corners=False)

        return out


# Configuration / default sizes for inputs and initialization
BATCH = 4
IN_CHANNELS = 48
DEPTH = 12
HEIGHT = 28
WIDTH = 20

POOL_PD = 3
POOL_PH = 2
POOL_PW = 1

HIDDEN_DIM = 128

def get_inputs() -> List[torch.Tensor]:
    """
    Create example input tensor of shape (BATCH, IN_CHANNELS, DEPTH, HEIGHT, WIDTH).
    """
    x = torch.randn(BATCH, IN_CHANNELS, DEPTH, HEIGHT, WIDTH)
    return [x]

def get_init_inputs() -> List:
    """
    Provide initialization arguments for Model:
      - in_channels
      - hidden_dim
      - pool_output tuple
    """
    return [IN_CHANNELS, HIDDEN_DIM, (POOL_PD, POOL_PH, POOL_PW)]