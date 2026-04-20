import torch
import torch.nn as nn
from typing import List

class Model(nn.Module):
    """
    Complex 3D-feature processing module that:
    - Applies channel-wise Dropout3d to 5D inputs (N, C, D, H, W)
    - Maps per-voxel channel features through a shared linear layer (nn.Linear)
    - Applies LeakyReLU non-linearity
    - Computes a global context vector via spatial average, transforms it and uses a sigmoid gating
      to re-scale per-voxel features (channel-wise)
    - Aggregates spatially (global average) and produces final outputs via a small linear classifier
    This combines nn.Dropout3d, nn.Linear, and nn.LeakyReLU in a non-trivial pattern.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_features: int,
        negative_slope: float = 0.01,
        dropout_p: float = 0.2,
    ):
        """
        Args:
            in_channels: Number of input channels (C).
            hidden_dim: Hidden feature dimension after the per-voxel linear transform.
            out_features: Number of output features (e.g., classes or regression dims).
            negative_slope: Negative slope for LeakyReLU.
            dropout_p: Dropout probability for Dropout3d (channel dropout).
        """
        super(Model, self).__init__()
        # Channel dropout across the spatial (D,H,W) map
        self.dropout3d = nn.Dropout3d(p=dropout_p)
        # Per-voxel linear mapping from in_channels -> hidden_dim (applied to last dim)
        self.linear1 = nn.Linear(in_channels, hidden_dim, bias=True)
        # Context transform used to produce gating (hidden_dim -> hidden_dim)
        self.context_linear = nn.Linear(hidden_dim, hidden_dim, bias=True)
        # Non-linearity
        self.activation = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        # Final classifier after spatial aggregation
        self.classifier = nn.Linear(hidden_dim, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (N, C, D, H, W)
        Returns:
            Tensor of shape (N, out_features)
        """
        # Step 1: channel-wise dropout (drops entire channels across spatial dims)
        x = self.dropout3d(x)  # (N, C, D, H, W)

        # Step 2: bring channels to the last dimension to apply Linear per-voxel
        # new shape: (N, D, H, W, C)
        x = x.permute(0, 2, 3, 4, 1)

        # Step 3: per-voxel linear transform (applied to last dim C -> hidden_dim)
        x = self.linear1(x)  # (N, D, H, W, hidden_dim)

        # Step 4: non-linearity
        x = self.activation(x)  # (N, D, H, W, hidden_dim)

        # Step 5: compute spatial global context (average over D,H,W)
        # shape -> (N, hidden_dim)
        context = x.mean(dim=(1, 2, 3))

        # Step 6: transform context and produce gating mask, then rescale per-voxel features
        gate = torch.sigmoid(self.context_linear(context))  # (N, hidden_dim)
        # reshape gate to broadcast over spatial dims: (N, 1, 1, 1, hidden_dim)
        gate = gate.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        x = x * gate  # gated per-voxel features

        # Step 7: spatial aggregation (global average) to get fixed-size per-sample vector
        x = x.mean(dim=(1, 2, 3))  # (N, hidden_dim)

        # Step 8: final classifier
        out = self.classifier(x)  # (N, out_features)

        return out

# Configuration / default sizes
batch_size = 8
in_channels = 16
depth = 4
height = 8
width = 8
hidden_dim = 64
out_features = 10
default_negative_slope = 0.02
default_dropout_p = 0.3

def get_inputs() -> List[torch.Tensor]:
    """
    Returns a list with a single 5D input tensor shaped (N, C, D, H, W).
    """
    x = torch.randn(batch_size, in_channels, depth, height, width, dtype=torch.float32)
    return [x]

def get_init_inputs() -> List:
    """
    Returns the initialization arguments for the Model constructor in order:
    [in_channels, hidden_dim, out_features, negative_slope, dropout_p]
    """
    return [in_channels, hidden_dim, out_features, default_negative_slope, default_dropout_p]