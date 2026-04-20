import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class Model(nn.Module):
    """
    Complex 3D channel-recalibration module that:
    - Applies ReflectionPad3d to the input volume
    - Pools each channel spatially to create a channel-sequence
    - Processes the channel-sequence with an LSTMCell (treating channels as time steps)
    - Applies AlphaDropout to the sequence of LSTM outputs
    - Computes per-channel gates via a small linear head and sigmoid
    - Reweights the original (padded) feature map per channel and returns the result cropped to original spatial size
    """
    def __init__(self, padding: int, hidden_dim: int, dropout_p: float = 0.1):
        """
        Args:
            padding (int): Reflection padding applied on all sides of the volume.
            hidden_dim (int): Hidden size for the LSTMCell.
            dropout_p (float): Dropout probability for AlphaDropout.
        """
        super(Model, self).__init__()
        self.padding = padding
        self.hidden_dim = hidden_dim
        self.pad_layer = nn.ReflectionPad3d(padding)
        # LSTMCell will process scalar inputs per time-step (per-channel pooled value)
        self.lstm_cell = nn.LSTMCell(input_size=1, hidden_size=hidden_dim)
        # Apply AlphaDropout to the sequence of LSTM outputs
        self.alpha_dropout = nn.AlphaDropout(dropout_p)
        # Map each LSTM output (per channel) to a single logit, then sigmoid to get gate
        self.gate_fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, D, H, W)
        
        Returns:
            torch.Tensor: Output tensor of same shape as input, with channels re-weighted.
        """
        B, C, D, H, W = x.shape

        # 1) Reflection pad the input
        x_padded = self.pad_layer(x) if self.padding > 0 else x  # shape (B, C, D+2p, H+2p, W+2p)

        # 2) Spatially pool each channel to form a (B, C) sequence (channel as time)
        #    Use mean pooling over spatial dims
        spatial_flat = x_padded.view(B, C, -1)
        channel_seq = spatial_flat.mean(dim=2)  # (B, C)

        # 3) Process the channel sequence with LSTMCell (channels as time steps)
        #    Each time step input is scalar per channel -> shape (B, 1)
        h = torch.zeros(B, self.hidden_dim, device=x.device, dtype=x.dtype)
        c = torch.zeros(B, self.hidden_dim, device=x.device, dtype=x.dtype)

        outputs: List[torch.Tensor] = []
        for t in range(C):
            inp_t = channel_seq[:, t].unsqueeze(1)  # (B, 1)
            h, c = self.lstm_cell(inp_t, (h, c))    # h: (B, hidden_dim)
            outputs.append(h)

        # Stack outputs -> (B, C, hidden_dim)
        seq_outputs = torch.stack(outputs, dim=1)

        # 4) Apply AlphaDropout to the sequence outputs
        seq_dropped = self.alpha_dropout(seq_outputs)

        # 5) Compute per-channel gates via a small linear projection + sigmoid
        #    gate_logits: (B, C, 1) -> squeeze -> (B, C)
        gate_logits = self.gate_fc(seq_dropped)  # (B, C, 1)
        gates = torch.sigmoid(gate_logits.squeeze(-1))  # (B, C)

        # 6) Reweight the padded feature map per channel and crop back to original spatial size
        #    Expand gates to spatial dims
        gates_expanded = gates.view(B, C, 1, 1, 1)
        x_reweighted = x_padded * gates_expanded  # shape (B, C, D+2p, H+2p, W+2p)

        if self.padding > 0:
            p = self.padding
            x_out = x_reweighted[..., p:-p, p:-p, p:-p]
        else:
            x_out = x_reweighted

        return x_out

# Configuration variables
batch_size = 8
channels = 12
depth = 16
height = 20
width = 20
padding = 2
hidden_dim = 32
dropout_p = 0.15

def get_inputs():
    """
    Returns a list with the input tensor matching the network input shape.
    """
    x = torch.randn(batch_size, channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters for the Model constructor:
    [padding, hidden_dim, dropout_p]
    """
    return [padding, hidden_dim, dropout_p]