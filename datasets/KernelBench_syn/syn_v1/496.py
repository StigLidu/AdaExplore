import torch
import torch.nn as nn

# Configuration
batch_size = 8
in_channels = 32
out_channels = 64
length = 512
pad = 3  # replication pad on each side

class Model(nn.Module):
    """
    A moderately complex 1D processing module that:
    - Applies replication padding along the temporal/spatial dimension.
    - Transposes to (B, L, C) to apply LayerNorm over channels.
    - Uses Hardswish non-linearity.
    - Performs a position-wise linear projection (per time step).
    - Computes a global channel summary from the original unpadded input,
      projects it to form a gating vector, and applies it to the sequence.
    - Reduces the sequence by mean pooling to produce a fixed-size output per batch.
    """
    def __init__(self, in_ch: int, out_ch: int, padding: int):
        super(Model, self).__init__()
        # Padding layer works on (B, C, L)
        self.pad = nn.ReplicationPad1d(padding)
        # LayerNorm over the channel dimension after we transpose to (B, L, C)
        self.layer_norm = nn.LayerNorm(in_ch)
        # Non-linearity
        self.act = nn.Hardswish()
        # Position-wise linear projection (applied to last dimension when input is (B, L, C))
        self.proj = nn.Linear(in_ch, out_ch, bias=True)
        # Gating projection: maps global per-channel summary to out_ch for multiplicative gating
        self.g_proj = nn.Linear(in_ch, out_ch, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, L)

        Returns:
            Tensor of shape (B, out_ch) obtained after padding, normalization, activation,
            projection, gating, and mean pooling across the (padded) length dimension.
        """
        # 1) Pad along the last dimension (replicating boundary values)
        x_padded = self.pad(x)  # shape: (B, C, L + 2*pad)

        # 2) Transpose to (B, L, C) so LayerNorm can normalize over channels
        x_t = x_padded.permute(0, 2, 1)  # shape: (B, L_p, C)

        # 3) Layer normalization over channel dimension
        x_ln = self.layer_norm(x_t)  # shape: (B, L_p, C)

        # 4) Non-linear activation
        x_act = self.act(x_ln)  # shape: (B, L_p, C)

        # 5) Position-wise linear projection to increase channels
        x_proj = self.proj(x_act)  # shape: (B, L_p, out_ch)

        # 6) Compute a global summary from the original (unpadded) input: mean over length
        #    This captures global channel statistics per batch sample.
        g = torch.mean(x, dim=2)  # shape: (B, C)

        # 7) Project the global summary to out_ch and produce a gate (use sigmoid for gating)
        g_out = self.g_proj(g)  # shape: (B, out_ch)
        gate = torch.sigmoid(g_out).unsqueeze(1)  # shape: (B, 1, out_ch), ready to broadcast

        # 8) Apply gating multiplicatively to the projected sequence
        gated = x_proj * gate  # shape: (B, L_p, out_ch)

        # 9) Aggregate across the (padded) length dimension to produce fixed-size output
        out = torch.mean(gated, dim=1)  # shape: (B, out_ch)

        return out

def get_inputs():
    """
    Returns the runtime inputs required by Model.forward:
    - A single tensor x of shape (batch_size, in_channels, length)
    """
    x = torch.randn(batch_size, in_channels, length)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters for the Model constructor:
    [in_channels, out_channels, pad]
    """
    return [in_channels, out_channels, pad]