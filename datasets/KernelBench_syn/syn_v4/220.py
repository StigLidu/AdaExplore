import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex sequence processing module that combines an LSTM encoder with an RNNCell-based
    iterative refinement stage and a ReLU6 activation. The model:
      - Encodes the input sequence with a multi-layer LSTM (batch_first=True).
      - Produces a fused summary from the LSTM outputs (mean pooling + last timestep).
      - Projects the fused summary into the RNNCell hidden space and uses it as the initial state.
      - Iteratively updates the hidden state across timesteps using an RNNCell whose inputs are
        projected LSTM time-step features passed through ReLU6.
      - Aggregates temporal states to compute a gating vector, applies the gate to the final
        RNN hidden state, and maps to the desired output dimension.
    """
    def __init__(
        self,
        input_size: int,
        lstm_hidden_size: int,
        lstm_layers: int,
        rnn_hidden_size: int,
        output_size: int,
    ):
        """
        Initializes the composite model.

        Args:
            input_size (int): Dimensionality of each input vector in the sequence.
            lstm_hidden_size (int): Hidden size of the LSTM encoder.
            lstm_layers (int): Number of stacked LSTM layers.
            rnn_hidden_size (int): Hidden size used by the RNNCell refinement stage.
            output_size (int): Final output dimensionality per batch entry.
        """
        super(Model, self).__init__()

        # LSTM encoder (encodes sequences)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
        )

        # Project LSTM hidden dims to RNNCell input/hidden dims
        self.init_proj = nn.Linear(lstm_hidden_size, rnn_hidden_size)
        self.step_proj = nn.Linear(lstm_hidden_size, rnn_hidden_size)

        # RNNCell iterative refinement
        self.rnn_cell = nn.RNNCell(rnn_hidden_size, rnn_hidden_size, nonlinearity="tanh")

        # Nonlinearity (ReLU6) applied in a couple of places
        self.relu6 = nn.ReLU6()

        # Final mapping to output dimension
        self.final = nn.Linear(rnn_hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        # x -> LSTM encoding
        # outputs: (batch, seq_len, lstm_hidden_size)
        outputs, (hn, cn) = self.lstm(x)

        # Temporal pooling: mean over time and take last timestep
        pooled = outputs.mean(dim=1)            # (batch, lstm_hidden_size)
        last_t = outputs[:, -1, :]              # (batch, lstm_hidden_size)

        # Fuse pooled and last timestep representations
        fused = 0.5 * (pooled + last_t)         # (batch, lstm_hidden_size)

        # Project fused representation into RNN hidden space and apply ReLU6
        rnn_init = self.relu6(self.init_proj(fused))  # (batch, rnn_hidden_size)

        # Iteratively refine hidden state across the sequence using RNNCell
        h = rnn_init
        states = []
        seq_len = outputs.size(1)
        for t in range(seq_len):
            # Project per-timestep LSTM features into RNNCell input space and apply ReLU6
            step_input = self.relu6(self.step_proj(outputs[:, t, :]))  # (batch, rnn_hidden_size)
            h = self.rnn_cell(step_input, h)  # (batch, rnn_hidden_size)
            states.append(h.unsqueeze(1))

        # Stack states to (batch, seq_len, rnn_hidden_size)
        states = torch.cat(states, dim=1)

        # Compute a temporal summary over the refined states and produce a gating vector
        temporal_summary = states.mean(dim=1)                      # (batch, rnn_hidden_size)
        gate = torch.sigmoid(temporal_summary + rnn_init)         # (batch, rnn_hidden_size)

        # Apply gate to the final hidden state and map to output
        gated_final = h * gate                                     # (batch, rnn_hidden_size)
        out = self.final(gated_final)                              # (batch, output_size)
        return out

# Configuration / default sizes
batch_size = 8
seq_len = 12
input_size = 64
lstm_hidden_size = 128
lstm_layers = 2
rnn_hidden_size = 128
output_size = 32

def get_inputs():
    """
    Generates a sample input sequence tensor with shape (batch_size, seq_len, input_size).
    """
    x = torch.randn(batch_size, seq_len, input_size)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters for the Model in the order:
    [input_size, lstm_hidden_size, lstm_layers, rnn_hidden_size, output_size]
    """
    return [input_size, lstm_hidden_size, lstm_layers, rnn_hidden_size, output_size]