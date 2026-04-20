import torch
import torch.nn as nn
from typing import List, Optional

"""
Complex sequence processing module that combines a multi-layer GRU, a per-timestep projection,
a lazy BatchNorm over the projected features, a recurrent refinement using GRUCell, and
a residual/skip connection from the original inputs. The module demonstrates mixing
nn.GRU, nn.GRUCell, and nn.LazyBatchNorm1d together in a non-trivial computation graph.
"""

# Module-level configuration
SEQ_LEN = 20
BATCH = 8
INPUT_DIM = 64
HIDDEN_SIZE = 128
NUM_LAYERS = 2
BIDIRECTIONAL = False
PROJ_DIM = 256
OUTPUT_DIM = 50  # Final output feature size


class Model(nn.Module):
    """
    Sequence model combining:
      - a multi-layer GRU to encode the input sequence
      - a linear projection of GRU outputs to a feature space (proj_dim)
      - a LazyBatchNorm1d applied across the flattened (batch * time) dimension
      - a GRUCell that iteratively refines projected features across time steps (stateful pass)
      - a skip/residual projection from input to proj_dim
      - temporal aggregation (mean over time) and a final linear projection to OUTPUT_DIM
    """

    def __init__(
        self,
        num_layers: int = NUM_LAYERS,
        hidden_size: int = HIDDEN_SIZE,
        bidirectional: bool = BIDIRECTIONAL,
        proj_dim: int = PROJ_DIM,
        input_dim: int = INPUT_DIM,
        output_dim: int = OUTPUT_DIM,
    ):
        super(Model, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.proj_dim = proj_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Core encoder: multi-layer GRU
        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
        )

        # Projection from GRU output features to a common projected feature dimension
        gru_out_dim = self.hidden_size * (2 if self.bidirectional else 1)
        self.proj = nn.Linear(gru_out_dim, self.proj_dim)

        # Lazy BatchNorm will initialize its weights on first forward based on incoming C dimension
        self.bn = nn.LazyBatchNorm1d()

        # GRUCell to refine the projected features in a stateful manner across time
        # Hidden size equals proj_dim so outputs can be directly used/residualed
        self.grucell = nn.GRUCell(input_size=self.proj_dim, hidden_size=self.proj_dim)

        # Skip connection: map original input_dim to proj_dim (applied per timestep)
        self.input_skip = nn.Linear(self.input_dim, self.proj_dim)

        # Final projection to desired output dimension
        self.final_proj = nn.Linear(self.proj_dim, self.output_dim)

        # Non-linearity
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input sequence of shape (seq_len, batch, input_dim).
            h0 (Optional[torch.Tensor]): Initial hidden state for GRU with shape
                                         (num_layers * num_directions, batch, hidden_size).
                                         If None, GRU will initialize zeros.

        Returns:
            torch.Tensor: Output tensor of shape (batch, output_dim).
        """
        # x -> (seq_len, batch, input_dim)
        seq_len, batch, in_dim = x.shape
        assert in_dim == self.input_dim, f"Expected input_dim={self.input_dim}, got {in_dim}"

        # Encode sequence with GRU
        # rnn_out: (seq_len, batch, gru_out_dim)
        rnn_out, h_n = self.gru(x, h0)  # h_n shape: (num_layers * num_directions, batch, hidden_size)

        # Project GRU outputs to proj_dim per timestep
        # rnn_out is (seq_len, batch, gru_out_dim) -> proj_seq (seq_len, batch, proj_dim)
        proj_seq = self.proj(rnn_out)

        # Apply BatchNorm across the feature dimension; BatchNorm1d expects (N, C) or (N, C, L).
        # We treat (batch * seq_len) as the batch dimension and proj_dim as channels.
        # Reshape: (seq_len, batch, proj_dim) -> (batch * seq_len, proj_dim)
        bn_input = proj_seq.permute(1, 0, 2).contiguous().view(batch * seq_len, self.proj_dim)
        bn_output = self.bn(bn_input)  # LazyBatchNorm1d will initialize on first call
        # Restore shape: (batch * seq_len, proj_dim) -> (seq_len, batch, proj_dim)
        bn_seq = bn_output.view(batch, seq_len, self.proj_dim).permute(1, 0, 2).contiguous()

        # Stateful refinement across time using GRUCell
        # Initialize cell hidden state to zeros on first device/type as bn_seq
        device = bn_seq.device
        cell_h = torch.zeros(batch, self.proj_dim, device=device, dtype=bn_seq.dtype)

        refined = []
        for t in range(seq_len):
            inp_t = bn_seq[t]  # (batch, proj_dim)
            # GRUCell takes (input, hidden) and returns new hidden
            cell_h = self.grucell(inp_t, cell_h)
            # Apply non-linearity to cell output
            refined_t = self.activation(cell_h)
            refined.append(refined_t)

        # Stack refined outputs: (seq_len, batch, proj_dim)
        refined_seq = torch.stack(refined, dim=0)

        # Compute skip/residual path from original inputs
        # Map original x (seq_len, batch, input_dim) -> skip_seq (seq_len, batch, proj_dim)
        # Linear operates on last dimension
        skip_seq = self.input_skip(x)

        # Combine refined features with skip using residual connection
        combined_seq = refined_seq + skip_seq  # (seq_len, batch, proj_dim)

        # Temporal aggregation: mean over time dimension
        aggregated = combined_seq.mean(dim=0)  # (batch, proj_dim)

        # Final projection
        out = self.final_proj(aggregated)  # (batch, output_dim)

        return out


def get_inputs() -> List[torch.Tensor]:
    """
    Returns a list of input tensors for the model forward:
      - x: random input sequence tensor shape (SEQ_LEN, BATCH, INPUT_DIM)
      - h0: initial hidden state for GRU with appropriate shape

    The shapes align with the module-level configuration variables.
    """
    x = torch.randn(SEQ_LEN, BATCH, INPUT_DIM)
    num_directions = 2 if BIDIRECTIONAL else 1
    h0 = torch.randn(NUM_LAYERS * num_directions, BATCH, HIDDEN_SIZE)
    return [x, h0]


def get_init_inputs() -> List:
    """
    Returns initialization parameters for the Model constructor in the following order:
      [num_layers, hidden_size, bidirectional, proj_dim, input_dim, output_dim]
    """
    return [NUM_LAYERS, HIDDEN_SIZE, BIDIRECTIONAL, PROJ_DIM, INPUT_DIM, OUTPUT_DIM]