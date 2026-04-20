import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

class Model(nn.Module):
    """
    Recurrent sequence model using an LSTMCell, AlphaDropout on the hidden state,
    a linear projection to outputs, and LogSoftmax to produce log-probabilities.

    The model processes an input sequence (batch, seq_len, input_dim) step-by-step:
      - At each timestep t, it runs an LSTMCell on x[:, t, :].
      - Applies AlphaDropout to the hidden state h_t.
      - Projects the dropped hidden state to output_dim logits.
      - Applies LogSoftmax over the feature dimension to produce log-probabilities.

    Returns a tensor of shape (batch, seq_len, output_dim) containing log-probabilities
    for each timestep.
    """
    def __init__(self, input_dim: int, hidden_size: int, output_dim: int, dropout_p: float = 0.1):
        """
        Initializes the recurrent model.

        Args:
            input_dim (int): Dimensionality of input features per timestep.
            hidden_size (int): Hidden size of the LSTMCell.
            output_dim (int): Number of output classes/features per timestep.
            dropout_p (float, optional): AlphaDropout probability. Default: 0.1
        """
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.dropout_p = dropout_p

        # Core recurrent cell
        self.lstm_cell = nn.LSTMCell(input_dim, hidden_size)

        # AlphaDropout to preserve self-normalizing properties (works well after SELU-like activations)
        self.alpha_dropout = nn.AlphaDropout(p=dropout_p)

        # Projection from hidden state to output logits
        self.output_proj = nn.Linear(hidden_size, output_dim)

        # LogSoftmax for numerically stable log-probabilities along feature dimension
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None, c0: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Processes the sequence and returns log-probabilities per timestep.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, input_dim).
            h0 (Optional[torch.Tensor]): Initial hidden state of shape (batch, hidden_size).
            c0 (Optional[torch.Tensor]): Initial cell state of shape (batch, hidden_size).

        Returns:
            torch.Tensor: Log-probabilities of shape (batch, seq_len, output_dim).
        """
        assert x.dim() == 3 and x.size(2) == self.input_dim, "Input must be (batch, seq_len, input_dim)"

        batch_size, seq_len, _ = x.size()
        device = x.device
        dtype = x.dtype

        # Initialize hidden and cell states if not provided
        if h0 is None:
            h = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        else:
            h = h0.to(device=device, dtype=dtype)
        if c0 is None:
            c = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        else:
            c = c0.to(device=device, dtype=dtype)

        outputs: List[torch.Tensor] = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, input_dim)

            # LSTM cell update
            h, c = self.lstm_cell(x_t, (h, c))  # both (batch, hidden_size)

            # Apply AlphaDropout to the hidden state (training behavior drops units)
            h_dropped = self.alpha_dropout(h)

            # Project to logits and compute log-probabilities
            logits = self.output_proj(h_dropped)  # (batch, output_dim)
            logp = self.log_softmax(logits)  # (batch, output_dim)

            outputs.append(logp)

        # Stack along time dimension -> shape (batch, seq_len, output_dim)
        out = torch.stack(outputs, dim=1)
        return out

# Module-level configuration variables
batch_size = 8
seq_len = 32
input_dim = 256
hidden_size = 512
output_dim = 1024
dropout_p = 0.05

def get_inputs():
    """
    Generates a random input sequence for testing.

    Returns:
        list: [x] where x has shape (batch_size, seq_len, input_dim).
    """
    x = torch.randn(batch_size, seq_len, input_dim)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for the Model constructor.

    Returns:
        list: [input_dim, hidden_size, output_dim, dropout_p]
    """
    return [input_dim, hidden_size, output_dim, dropout_p]