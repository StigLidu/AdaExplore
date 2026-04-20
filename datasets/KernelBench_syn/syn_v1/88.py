import torch
import torch.nn as nn
from typing import List

# Configuration
batch_size = 8
seq_len = 12
channels = 64
height = 16
width = 16
hidden_size = 128
output_dim = 10

class Model(nn.Module):
    """
    Complex module that processes a small sequence of feature maps per batch item.
    Pipeline:
      - Apply ConstantPad2d to each frame (spatial padding)
      - Move channels to last dim and apply RMSNorm over channels
      - Spatially pool to obtain a per-frame feature vector
      - Run a GRUCell recurrently over the sequence of frame vectors
      - Combine final hidden state with a projected global context and produce outputs via a linear head

    Input expected shape: (batch, seq_len, channels, height, width)
    Output shape: (batch, output_dim)
    """
    def __init__(self,
                 in_channels: int = channels,
                 hid_size: int = hidden_size,
                 out_dim: int = output_dim,
                 pad: tuple = (1, 1, 2, 2),
                 pad_value: float = 0.0):
        super(Model, self).__init__()
        # Spatial pad: (left, right, top, bottom)
        self.pad2d = nn.ConstantPad2d(pad, pad_value)
        # RMSNorm applied over channels -> move channels to last dim before calling
        self.rms = nn.RMSNorm(in_channels, eps=1e-6)
        # GRU cell: operates on per-frame pooled channel vectors
        self.gru_cell = nn.GRUCell(input_size=in_channels, hidden_size=hid_size)
        # Project global context (pooled over time) into hidden space to form a residual/context vector
        self.context_proj = nn.Linear(in_channels, hid_size)
        # Final classification/regression head
        self.head = nn.Linear(hid_size, out_dim)
        # Small gating layer to combine GRU hidden and context
        self.gate = nn.Linear(hid_size * 2, hid_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (B, output_dim)
        """
        B, T, C, H, W = x.shape
        # Merge batch and time for spatial ops
        x_bt = x.view(B * T, C, H, W)                           # (B*T, C, H, W)
        # 1) Spatial padding
        x_padded = self.pad2d(x_bt)                            # (B*T, C, H', W')
        Hp, Wp = x_padded.shape[-2], x_padded.shape[-1]

        # 2) Move channels to last dim to apply RMSNorm over channel dimension
        x_perm = x_padded.permute(0, 2, 3, 1).contiguous()     # (B*T, H', W', C)
        x_normed = self.rms(x_perm)                            # RMSNorm normalizes last dim (C)
        # 3) Spatial pooling to get per-frame vector
        # Use mean over H' and W' -> (B*T, C)
        frame_feats = x_normed.mean(dim=(1, 2))                # (B*T, C)

        # Restore (B, T, C) sequence layout
        seq_feats = frame_feats.view(B, T, C)                  # (B, T, C)

        # Precompute a global context vector per batch by averaging frame features
        global_context = seq_feats.mean(dim=1)                 # (B, C)
        ctx_proj = self.context_proj(global_context)           # (B, hid_size)

        # 4) Recurrent processing across time using GRUCell
        h = torch.zeros(B, self.gru_cell.hidden_size, device=x.device, dtype=x.dtype)
        for t in range(T):
            inp_t = seq_feats[:, t, :]                         # (B, C)
            # GRUCell expects (B, input_size). Here input_size == C
            h = self.gru_cell(inp_t, h)                        # (B, hid_size)

        # 5) Combine GRU final state with projected context via a learned gate
        combined = torch.cat([h, ctx_proj], dim=1)             # (B, hid_size*2)
        gate = torch.sigmoid(self.gate(combined))              # (B, hid_size)
        fused = gate * h + (1.0 - gate) * ctx_proj             # (B, hid_size)

        # 6) Final head
        out = self.head(fused)                                 # (B, output_dim)
        return out

def get_inputs() -> List[torch.Tensor]:
    # Create a random batch of sequences of feature maps
    x = torch.randn(batch_size, seq_len, channels, height, width)
    return [x]

def get_init_inputs() -> List:
    # No special external initialization tensors required; model constructs its own hidden state.
    return []