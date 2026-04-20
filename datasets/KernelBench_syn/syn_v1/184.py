import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

class Model(nn.Module):
    """
    Complex model that:
    - Applies AdaptiveMaxPool2d to spatial inputs
    - Uses Tanhshrink activation on the pooled features
    - Pads the sequence dimension with ReplicationPad1d
    - Projects channel features into a hidden embedding per timestep via nn.Linear
    - Aggregates the sequence and produces a final projection

    This creates a multi-stage pipeline combining pooling, non-linearity, padding,
    and learned linear projections.
    """
    def __init__(
        self,
        in_channels: int,
        pooled_size: Tuple[int, int],
        pad: Tuple[int, int],
        hidden_dim: int,
        out_dim: int
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            pooled_size (Tuple[int,int]): Target (H, W) for AdaptiveMaxPool2d.
            pad (Tuple[int,int]): (left_pad, right_pad) for ReplicationPad1d.
            hidden_dim (int): Feature dimension after per-timestep projection.
            out_dim (int): Final output dimension after aggregation.
        """
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.pooled_size = pooled_size
        self.pad = pad
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        # Adaptive spatial pooling to produce a fixed-size spatial grid
        self.pool = nn.AdaptiveMaxPool2d(output_size=self.pooled_size)

        # Element-wise non-linearity that subtracts tanh(x)
        self.tanhshrink = nn.Tanhshrink()

        # Replication pad for sequence length expansion along the temporal/sequence axis
        self.pad1d = nn.ReplicationPad1d(self.pad)

        # Project per-timestep channel vector into hidden embedding
        # Input features = in_channels, since we'll treat channels as features per timestep
        self.proj = nn.Linear(self.in_channels, self.hidden_dim, bias=True)

        # Final projection from aggregated sequence embedding to desired output dim
        self.final_proj = nn.Linear(self.hidden_dim, self.out_dim, bias=True)

        # Optional small normalization on the aggregated embedding for stability
        self.norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        1. x: (B, C, H, W)
        2. pooled -> (B, C, Hp, Wp)
        3. reshape to sequence (B, C, L) where L = Hp * Wp
        4. tanhshrink activation -> (B, C, L)
        5. replication pad along L -> (B, C, L_padded)
        6. permute to (B, L_padded, C) and project channels -> (B, L_padded, hidden_dim)
        7. sequence aggregation (mean + max concat) -> (B, hidden_dim)
        8. normalize and final projection -> (B, out_dim)
        """
        # 1 -> 2
        pooled = self.pool(x)  # (B, C, Hp, Wp)

        B, C, Hp, Wp = pooled.shape
        L = Hp * Wp

        # 3 reshape to sequence
        seq = pooled.view(B, C, L)  # (B, C, L)

        # 4 tanhshrink activation
        seq = self.tanhshrink(seq)  # (B, C, L)

        # 5 replication pad along the length dimension
        seq_padded = self.pad1d(seq)  # (B, C, L_padded)
        L_padded = seq_padded.size(-1)

        # 6 permute to (B, L_padded, C) then project channels -> embedding per timestep
        seq_per_t = seq_padded.permute(0, 2, 1).contiguous()  # (B, L_padded, C)
        emb = self.proj(seq_per_t)  # (B, L_padded, hidden_dim)

        # 7 aggregation: combine mean and max over the sequence for richer summary
        mean_pool = emb.mean(dim=1)  # (B, hidden_dim)
        max_pool, _ = emb.max(dim=1)  # (B, hidden_dim)
        agg = 0.6 * mean_pool + 0.4 * max_pool  # weighted fusion (B, hidden_dim)

        # 8 normalize and final projection
        agg = self.norm(agg)  # (B, hidden_dim)
        out = self.final_proj(agg)  # (B, out_dim)

        return out

# Module-level configuration variables (example sizes)
batch_size = 8
in_channels = 16
height = 64
width = 48

pooled_H = 8
pooled_W = 6

pad_left = 2
pad_right = 3

hidden_dim = 128
out_dim = 64

def get_inputs():
    """
    Returns:
        List containing a single input tensor of shape (batch_size, in_channels, height, width)
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization arguments for Model:
    [in_channels, pooled_size_tuple, pad_tuple, hidden_dim, out_dim]
    """
    return [
        in_channels,
        (pooled_H, pooled_W),
        (pad_left, pad_right),
        hidden_dim,
        out_dim
    ]