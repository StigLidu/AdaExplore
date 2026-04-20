import torch
import torch.nn as nn
from typing import List

class Model(nn.Module):
    """
    Sequence processor that combines an nn.RNNCell with multiple projection and gating layers.

    For each time step:
      - Updates hidden state with an RNNCell using the current input and previous hidden.
      - Projects both the input and hidden state into a shared projection dimension.
      - Computes a learned gate (sigmoid) from the concatenated raw input and hidden state to blend projections.
      - Applies a residual projection + non-linearity to produce the per-time output.
    The module returns the sequence of per-time outputs with shape (batch, seq_len, proj_dim).
    """
    def __init__(self, input_dim: int, hidden_dim: int, proj_dim: int):
        """
        Args:
            input_dim: Dimensionality of input features per time step.
            hidden_dim: Hidden size used by the RNNCell.
            proj_dim: Dimension for intermediate projections and final outputs.
        """
        super(Model, self).__init__()
        # Recurrent cell that updates hidden state per time step
        self.rnn_cell = nn.RNNCell(input_size=input_dim, hidden_size=hidden_dim, nonlinearity='tanh')

        # Project raw input into projection space
        self.input_proj = nn.Linear(input_dim, proj_dim)

        # Project hidden state into projection space
        self.hidden_proj = nn.Linear(hidden_dim, proj_dim)

        # Gate that computes blending coefficients from [input, hidden] -> proj_dim
        self.gate = nn.Linear(input_dim + hidden_dim, proj_dim)

        # Final residual projection applied to the blended projection
        self.residual = nn.Linear(proj_dim, proj_dim)

        # Optional small output bias layer (identity-style mapping with capability to learn)
        self.output_bias = nn.Linear(proj_dim, proj_dim, bias=True)

        # Activation used after residual
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes a batch of sequences.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            Tensor of shape (batch, seq_len, proj_dim) - per-time outputs.
        """
        batch_size, seq_len, input_dim = x.size()
        # Initialize hidden state to zeros
        h = x.new_zeros(batch_size, self.rnn_cell.hidden_size)

        outputs: List[torch.Tensor] = []
        for t in range(seq_len):
            x_t = x[:, t, :]                           # (batch, input_dim)
            # 1) RNNCell update: combine x_t and previous hidden to produce new hidden
            h = self.rnn_cell(x_t, h)                 # (batch, hidden_dim)

            # 2) Project input and hidden into the same projection dimension
            x_proj = self.input_proj(x_t)             # (batch, proj_dim)
            h_proj = self.hidden_proj(h)              # (batch, proj_dim)

            # 3) Compute gate from raw (x_t, h) to determine blending proportions
            gate_input = torch.cat([x_t, h], dim=1)   # (batch, input_dim + hidden_dim)
            gate = torch.sigmoid(self.gate(gate_input))  # (batch, proj_dim)

            # 4) Blend projections using the gate
            blended = gate * h_proj + (1.0 - gate) * x_proj  # (batch, proj_dim)

            # 5) Residual connection + non-linearity and a small output bias transform
            res = self.residual(blended)               # (batch, proj_dim)
            out_t = self.activation(res + blended)     # (batch, proj_dim)
            out_t = self.output_bias(out_t)            # (batch, proj_dim)

            outputs.append(out_t)

        # Stack time dimension back
        return torch.stack(outputs, dim=1)  # (batch, seq_len, proj_dim)


# Module-level configuration variables
batch_size = 32
seq_len = 20
input_dim = 64
hidden_dim = 128
proj_dim = 64

def get_inputs():
    """
    Returns:
        List containing a single input tensor of shape (batch_size, seq_len, input_dim).
    """
    x = torch.randn(batch_size, seq_len, input_dim, dtype=torch.float32)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for constructing the Model:
      [input_dim, hidden_dim, proj_dim]
    """
    return [input_dim, hidden_dim, proj_dim]