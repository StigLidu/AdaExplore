import torch
import torch.nn as nn
from typing import Optional, List

class Model(nn.Module):
    """
    Sequence processing model that combines a GRUCell recurrence with
    SiLU non-linearity and a sigmoid gating mechanism to produce a
    compact sequence representation.

    Computation pattern (per timestep):
      1. Update hidden state with GRUCell using current input and previous hidden.
      2. Compute a residual transformed input (Linear -> SiLU).
      3. Combine GRU hidden and transformed input, project to output space.
      4. Compute a sigmoid gate from hidden and modulate the projected output.
      5. Collect per-timestep outputs and aggregate across the sequence.

    Final output: aggregated (mean) per-timestep modulated projection -> (batch, output_size)
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initializes layers used in the model.

        Args:
            input_size (int): Dimensionality of input features per timestep.
            hidden_size (int): Dimensionality of the GRU hidden state.
            output_size (int): Dimensionality of the per-timestep output projection.
        """
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # GRU cell for recurrent state update
        self.gru_cell = nn.GRUCell(input_size=input_size, hidden_size=hidden_size)

        # Transformations for the input and hidden -> output projection
        self.input_transform = nn.Linear(input_size, hidden_size, bias=True)
        self.output_proj = nn.Linear(hidden_size, output_size, bias=True)

        # Gate generator from hidden state to modulate outputs
        self.gate_proj = nn.Linear(hidden_size, output_size, bias=True)

        # Non-linearities from the provided set
        self.act = nn.SiLU()
        self.gate_act = nn.Sigmoid()

    def forward(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Processes an input sequence and returns an aggregated output.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, input_size).
            h0 (Optional[torch.Tensor]): Optional initial hidden state of shape (batch, hidden_size).
                                         If None, initialized to zeros.

        Returns:
            torch.Tensor: Aggregated output of shape (batch, output_size).
        """
        batch, seq_len, inp_dim = x.shape
        assert inp_dim == self.input_size, f"Expected input_size {self.input_size}, got {inp_dim}"

        if h0 is None:
            h = x.new_zeros(batch, self.hidden_size)
        else:
            h = h0

        # Collect per-timestep outputs
        outputs: List[torch.Tensor] = []

        for t in range(seq_len):
            xt = x[:, t, :]  # (batch, input_size)

            # 1) GRUCell update: new hidden (depends on xt and previous h)
            h = self.gru_cell(xt, h)  # (batch, hidden_size)

            # 2) Residual transformed input passed through SiLU
            transformed_in = self.act(self.input_transform(xt))  # (batch, hidden_size)

            # 3) Combine hidden and transformed input (element-wise add) and project
            combined = h + transformed_in  # (batch, hidden_size)
            proj_out = self.output_proj(combined)  # (batch, output_size)

            # 4) Compute gate from hidden to modulate projection
            gate = self.gate_act(self.gate_proj(h))  # (batch, output_size)
            gated_out = proj_out * gate  # (batch, output_size)

            outputs.append(gated_out)

        # Stack and aggregate across time dimension (mean pooling)
        stacked = torch.stack(outputs, dim=1)  # (batch, seq_len, output_size)
        aggregated = stacked.mean(dim=1)  # (batch, output_size)

        return aggregated

# Configuration variables
batch_size = 8
seq_len = 20
input_size = 128
hidden_size = 256
output_size = 64

def get_inputs():
    """
    Returns inputs for the forward pass:
      - x: random sequence tensor (batch_size, seq_len, input_size)
      - h0: optional random initial hidden state (batch_size, hidden_size)

    The model's forward signature accepts (x, h0), so both are returned.
    """
    x = torch.randn(batch_size, seq_len, input_size)
    h0 = torch.randn(batch_size, hidden_size)
    return [x, h0]

def get_init_inputs():
    """
    Returns the initialization parameters for the Model constructor:
      [input_size, hidden_size, output_size]
    """
    return [input_size, hidden_size, output_size]