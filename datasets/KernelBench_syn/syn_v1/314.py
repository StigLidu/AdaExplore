import torch
import torch.nn as nn
from typing import List

"""
Complex sequence processing module that combines:
- An RNNCell unrolled across time to produce a sequence of hidden states
- Projection of RNN hidden states into a transformer-compatible d_model using LazyLinear
- A TransformerEncoderLayer applied to the projected sequence
- Residual connection + LayerNorm, sequence pooling, and a final LazyLinear projection back
This demonstrates use of nn.RNNCell, nn.TransformerEncoderLayer, and nn.LazyLinear together.
"""

# Configuration (module-level)
batch_size = 8
seq_len = 20
input_size = 32
hidden_size = 64
d_model = 128  # must be divisible by nhead
nhead = 8
dim_feedforward = 256
dropout = 0.1

class Model(nn.Module):
    """
    Sequence model that first processes inputs step-by-step with an RNNCell,
    then feeds the sequence of hidden states into a TransformerEncoderLayer after
    projecting to d_model using LazyLinear. The transformer output is combined
    with a residual connection, normalized, pooled across time, and finally projected
    back to a desired output dimension via another LazyLinear.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super(Model, self).__init__()
        # Recurrent cell that will be unrolled manually across the sequence dimension
        self.rnn_cell = nn.RNNCell(input_size=input_size, hidden_size=hidden_size, nonlinearity='tanh')
        self.hidden_size = hidden_size

        # LazyLinear will infer in_features on first forward call
        # Project RNN hidden vectors to transformer d_model
        self.project_to_dmodel = nn.LazyLinear(out_features=d_model)

        # Single transformer encoder layer (contains self-attention + feedforward)
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu'
        )

        # LayerNorm over the last dimension (d_model)
        self.norm = nn.LayerNorm(d_model)

        # Final projection from pooled d_model back to a desired output dimension (here equal to input_size)
        self.final_proj = nn.LazyLinear(out_features=input_size)

        # Small dropout for regularization
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, input_size)
        """
        # Validate input dims
        if x.dim() != 3:
            raise ValueError("Expected input of shape (batch, seq_len, input_size)")

        bsz, seq, _ = x.size()

        # Initialize hidden state for RNNCell
        h_t = x.new_zeros(bsz, self.hidden_size)

        # Unroll RNNCell across time, collecting hidden states
        hidden_states: List[torch.Tensor] = []
        for t in range(seq):
            x_t = x[:, t, :]           # (batch, input_size)
            h_t = self.rnn_cell(x_t, h_t)  # (batch, hidden_size)
            hidden_states.append(h_t)

        # Stack to shape (seq, batch, hidden_size) as expected by TransformerEncoderLayer (seq first)
        hs = torch.stack(hidden_states, dim=0)  # (seq, batch, hidden_size)

        # Project hidden states to d_model for transformer processing
        proj = self.project_to_dmodel(hs)  # (seq, batch, d_model)

        # Apply transformer encoder layer (self-attention + feedforward)
        trans_out = self.transformer_layer(proj)  # (seq, batch, d_model)

        # Residual connection and normalization
        out = self.norm(trans_out + proj)  # (seq, batch, d_model)
        out = self.dropout(out)

        # Pool across time dimension (mean pooling)
        pooled = out.mean(dim=0)  # (batch, d_model)

        # Final projection back to input_size (or desired output size)
        recon = self.final_proj(pooled)  # (batch, input_size)

        # Non-linear output activation
        return torch.tanh(recon)


def get_inputs():
    """
    Returns:
        list containing a single input tensor shaped (batch_size, seq_len, input_size)
    """
    x = torch.randn(batch_size, seq_len, input_size)
    return [x]


def get_init_inputs():
    """
    Returns initialization parameters for the Model constructor in order:
    (input_size, hidden_size, d_model, nhead, dim_feedforward, dropout)
    """
    return [input_size, hidden_size, d_model, nhead, dim_feedforward, dropout]