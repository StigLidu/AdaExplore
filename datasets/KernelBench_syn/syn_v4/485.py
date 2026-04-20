import torch
import torch.nn as nn
from typing import List, Any

class Model(nn.Module):
    """
    Complex sequence processing model that:
    - projects input tokens into an RNN feature space,
    - runs a multi-layer (optionally bidirectional) Elman RNN,
    - computes a learned mixture between the RNN output and a Tanhshrink residual,
    - uses GELU inside the gating mechanism,
    - pools across the time dimension and projects back to a target feature size.

    This model demonstrates combining nn.RNN, nn.GELU, and nn.Tanhshrink
    in a short computation graph with residual mixing and learned gating.
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, bidirectional: bool = False):
        """
        Initializes the model.

        Args:
            input_size (int): Number of input features per time step.
            hidden_size (int): Hidden size per RNN direction.
            num_layers (int): Number of stacked RNN layers.
            bidirectional (bool): Whether the RNN is bidirectional.
        """
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Number of directions for the RNN (1 or 2)
        self.num_directions = 2 if bidirectional else 1
        # Feature dimension after RNN (hidden per direction * directions)
        self.rnn_feature_dim = hidden_size * self.num_directions

        # Project raw inputs into the RNN input space
        self.input_proj = nn.Linear(input_size, self.rnn_feature_dim)

        # Multi-layer Elman RNN (tanh nonlinearity)
        self.rnn = nn.RNN(
            input_size=self.rnn_feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity='tanh',
            batch_first=True,
            bidirectional=bidirectional
        )

        # Gating network: compute per-feature gates from RNN outputs
        self.gate_lin = nn.Linear(self.rnn_feature_dim, self.rnn_feature_dim)
        self.gelu = nn.GELU()           # Activation used inside the gate
        self.tanhshrink = nn.Tanhshrink()  # Used to compute a residual-like correction

        # Final projection after temporal pooling back to input feature size
        self.output_proj = nn.Linear(self.rnn_feature_dim, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Steps:
            1. Project input features to RNN feature dimension.
            2. Run through the RNN to get sequence outputs.
            3. Build a residual via Tanhshrink between projected input and RNN output.
            4. Compute a GELU-activated gating signal (squashed by sigmoid).
            5. Mix RNN output and residual with the gate.
            6. Pool across sequence dimension (mean) and project to output size.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch, input_size).
        """
        # 1) Project inputs -> (batch, seq_len, rnn_feature_dim)
        proj_in = self.input_proj(x)

        # 2) RNN processing -> (batch, seq_len, rnn_feature_dim)
        # Note: rnn expects input_size == proj_in.size(-1)
        rnn_out, _ = self.rnn(proj_in)

        # 3) Residual-like correction using Tanhshrink applied elementwise
        #    between projected input and RNN output
        residual = self.tanhshrink(proj_in - rnn_out)

        # 4) Compute gating values per feature: Linear -> GELU -> sigmoid
        gate_pre = self.gate_lin(rnn_out)
        gate_act = self.gelu(gate_pre)
        gate = torch.sigmoid(gate_act)  # (batch, seq_len, rnn_feature_dim)

        # 5) Mix RNN output and residual using the gate
        mixed = gate * rnn_out + (1.0 - gate) * residual  # (batch, seq_len, rnn_feature_dim)

        # 6) Pool across time (mean) and project back to input feature dimension
        pooled = mixed.mean(dim=1)  # (batch, rnn_feature_dim)
        out = self.output_proj(pooled)  # (batch, input_size)

        return out


# Module-level configuration variables
batch_size = 8
seq_len = 20
input_size = 64
hidden_size = 128
num_layers = 2
bidirectional = False

def get_inputs() -> List[torch.Tensor]:
    """
    Returns a list containing a single input tensor for the model:
    shape (batch_size, seq_len, input_size).
    """
    x = torch.randn(batch_size, seq_len, input_size)
    return [x]

def get_init_inputs() -> List[Any]:
    """
    Returns the initialization parameters for constructing the Model:
    [input_size, hidden_size, num_layers, bidirectional]
    """
    return [input_size, hidden_size, num_layers, bidirectional]