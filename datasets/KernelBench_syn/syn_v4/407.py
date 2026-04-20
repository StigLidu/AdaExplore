import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Any

class Model(nn.Module):
    """
    A composite 1D sequence processing module that demonstrates a small attention-like
    aggregation using Circular Padding, a convolutional feature extractor, a Tanh nonlinearity,
    and a Softmax-based temporal weighting. The aggregated context is projected back to the
    input channel dimension and used to modulate the per-channel temporal averages.

    Computation steps (high level):
    1. Circularly pad the input along the temporal dimension.
    2. Apply a Conv1d to extract local features.
    3. Apply Tanh activation to the conv features.
    4. Compute temporal attention weights by averaging features across channels and applying Softmax.
    5. Produce a context vector via weighted-sum over time of the conv features.
    6. Project context to input channel dimension and apply Tanh gating.
    7. Modulate the per-channel temporal averages of the original input with the gate and return.
    """
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int, circular_padding: int):
        """
        Initializes the composite module.

        Args:
            in_channels (int): Number of input channels.
            hidden_channels (int): Number of channels produced by the convolutional feature extractor.
            kernel_size (int): Size of the 1D convolution kernel (local context).
            circular_padding (int): Amount of circular padding applied on each side of the sequence.
        """
        super(Model, self).__init__()
        # Circular padding layer (pads last dimension)
        self.pad = nn.CircularPad1d(circular_padding)
        # Conv1d extracts local features from the padded sequence
        # We set bias=True to allow flexible affine mapping in conv
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=kernel_size, bias=True)
        # Nonlinearities
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        # Project aggregated context back to input channel dimension
        self.project = nn.Linear(hidden_channels, in_channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, in_channels), representing
                          gated per-channel temporal summaries.
        """
        # x: (B, C_in, L)
        # 1) Circular pad along temporal dimension -> (B, C_in, L + 2*pad)
        x_padded = self.pad(x)

        # 2) Local feature extraction via Conv1d -> (B, C_hidden, L_out)
        feat = self.conv(x_padded)

        # 3) Nonlinearity
        feat_act = self.tanh(feat)  # (B, C_hidden, L_out)

        # 4) Compute temporal attention weights:
        #    First collapse channel dimension by mean to get a temporal score map,
        #    then apply Softmax across the temporal dimension to obtain weights.
        scores = feat_act.mean(dim=1, keepdim=True)  # (B, 1, L_out)
        attn_weights = self.softmax(scores)          # (B, 1, L_out)

        # 5) Weighted temporal aggregation to produce a context vector per sample and hidden channel
        #    Multiply broadcasting attn_weights across channels then sum over time.
        context = (feat * attn_weights).sum(dim=2)   # (B, C_hidden)

        # 6) Project context back to input channel dimension and apply Tanh gating
        gate = self.tanh(self.project(context))      # (B, C_in)

        # 7) Compute per-channel temporal average of the original (unpadded) input
        x_avg = x.mean(dim=2)                        # (B, C_in)

        # 8) Modulate the averages with the gate (residual-style: x_avg * (1 + gate))
        out = x_avg * (1.0 + gate)                   # (B, C_in)

        return out

# Module-level configuration (example values)
batch_size = 8
in_channels = 16
hidden_channels = 32
seq_len = 128
kernel_size = 5
circular_padding = 2  # chosen so that seq length is preserved after conv with this kernel

def get_inputs() -> List[torch.Tensor]:
    """
    Returns a list containing the input tensor to the Model.
    Shape: (batch_size, in_channels, seq_len)
    """
    x = torch.randn(batch_size, in_channels, seq_len)
    return [x]

def get_init_inputs() -> List[Any]:
    """
    Returns the initialization parameters for the Model:
    [in_channels, hidden_channels, kernel_size, circular_padding]
    """
    return [in_channels, hidden_channels, kernel_size, circular_padding]