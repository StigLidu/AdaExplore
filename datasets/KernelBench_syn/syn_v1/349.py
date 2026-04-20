import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Any

# Configuration variables
batch_size = 8
channels = 3
seq_len = 128
pad_size = 3
input_proj_dim = 64
gru_hidden = 128
gru_layers = 2
bidirectional = True

class Model(nn.Module):
    """
    A composite model that demonstrates a sequence processing pipeline combining:
      - 1D replication padding
      - Lazy instance normalization applied in a 4D context
      - Per-timestep linear projection
      - Multi-layer (optionally bidirectional) GRU
      - Output projection back to original channel dimensionality

    Input shape: (batch, channels, seq_len)
    Output shape: (batch, channels, seq_len)  -- same temporal length as input (padding is removed)
    """
    def __init__(
        self,
        in_channels: int,
        input_proj_dim: int,
        gru_hidden: int,
        gru_layers: int = 2,
        pad: int = 3,
        bidirectional: bool = True,
    ):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.pad = pad
        self.bidirectional = bidirectional

        # Replication padding for temporal boundaries
        self.pad_layer = nn.ReplicationPad1d(self.pad)

        # Lazy InstanceNorm2d: will infer num_features (channels) on first forward
        # We add a dummy spatial dimension before/after normalization.
        self.inst_norm = nn.LazyInstanceNorm2d()

        # Per-timestep projection to reduce/expand features before GRU
        # We'll apply this as a linear on the last dimension (features)
        self.input_proj = nn.Linear(in_channels, input_proj_dim)

        # GRU backbone
        self.gru = nn.GRU(
            input_size=input_proj_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

        # Project GRU outputs back to original channel dimensionality
        gru_out_dim = gru_hidden * (2 if self.bidirectional else 1)
        self.output_proj = nn.Linear(gru_out_dim, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, L)

        Returns:
            torch.Tensor: Output tensor of shape (B, C, L) (original temporal length)
        """
        # x: (B, C, L)
        # Step 1: Replication padding along the temporal dimension
        x_padded = self.pad_layer(x)  # (B, C, L + 2*pad)

        # Step 2: Instance normalization requires 4D input (N, C, H, W)
        # Insert a dummy height dimension of 1 -> (B, C, 1, Lp)
        x_4d = x_padded.unsqueeze(2)
        x_norm = self.inst_norm(x_4d)
        # Remove dummy height dimension -> (B, C, Lp)
        x_norm = x_norm.squeeze(2)

        # Step 3: Prepare sequence for per-timestep linear + GRU
        # Permute to (B, Lp, C) so linear and GRU operate on feature dim
        x_seq = x_norm.permute(0, 2, 1)

        # Step 4: Per-timestep projection
        x_proj = self.input_proj(x_seq)  # (B, Lp, input_proj_dim)

        # Step 5: GRU processing
        gru_out, _ = self.gru(x_proj)  # (B, Lp, gru_out_dim)

        # Step 6: Project GRU outputs back to channel space
        out_seq = self.output_proj(gru_out)  # (B, Lp, C)

        # Permute back to (B, C, Lp)
        out_padded = out_seq.permute(0, 2, 1)

        # Step 7: Remove padding and return original length sequence
        if self.pad > 0:
            out = out_padded[:, :, self.pad:-self.pad]
        else:
            out = out_padded

        return out

def get_inputs() -> List[torch.Tensor]:
    """
    Generates a sample input tensor consistent with the configuration variables.
    Returns a list so it matches the example interface.
    """
    x = torch.randn(batch_size, channels, seq_len)
    return [x]

def get_init_inputs() -> List[Any]:
    """
    Returns the initialization parameters that can be used to construct the Model.
    This mirrors the constructor signature:
      Model(in_channels, input_proj_dim, gru_hidden, gru_layers, pad, bidirectional)
    """
    return [channels, input_proj_dim, gru_hidden, gru_layers, pad_size, bidirectional]