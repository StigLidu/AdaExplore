import torch
import torch.nn as nn

class Model(nn.Module):
    """
    RNN-based sequence encoder that applies:
      - a multi-layer nn.RNN over the input sequence (batch_first=True)
      - channel-wise Dropout1d across the hidden feature channels
      - a sigmoid gating computed from sequence-average pooled features
      - a gated blend of the final time-step and the sequence-average
    This combines recurrent, dropout and sigmoid operations to create a
    compact, gated sequence representation.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: str = "tanh",
        dropout_p: float = 0.0,
        bidirectional: bool = False,
    ):
        """
        Args:
            input_size (int): Dimensionality of input features.
            hidden_size (int): Hidden size of the RNN.
            num_layers (int): Number of stacked RNN layers.
            nonlinearity (str): 'tanh' or 'relu' nonlinearity for nn.RNN.
            dropout_p (float): Drop probability for Dropout1d and RNN (when num_layers>1).
            bidirectional (bool): If True, use a bidirectional RNN.
        """
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.dropout_p = dropout_p
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            dropout=dropout_p if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        # Dropout1d applied over the feature/channel dimension (C) for sequence data
        # We will permute (batch, seq_len, channels) -> (batch, channels, seq_len)
        self.dropout1d = nn.Dropout1d(dropout_p)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, h0: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass producing a gated sequence embedding.

        Args:
            x (torch.Tensor): Input sequence of shape (batch, seq_len, input_size).
            h0 (torch.Tensor, optional): Initial hidden state of shape
                (num_layers * num_directions, batch, hidden_size). If None, RNN initializes zeros.

        Returns:
            torch.Tensor: Gated embedding of shape (batch, channels) where channels = hidden_size * num_directions.
        """
        # Apply the RNN. rnn_out: (batch, seq_len, channels)
        rnn_out, h_n = self.rnn(x, h0) if h0 is not None else self.rnn(x)

        # Apply channel-wise dropout: permute to (batch, channels, seq_len) for Dropout1d
        # then back to (batch, seq_len, channels)
        rnn_out_perm = rnn_out.permute(0, 2, 1)  # (B, C, L)
        dropped = self.dropout1d(rnn_out_perm)
        dropped = dropped.permute(0, 2, 1)  # (B, L, C)

        # Compute sequence-average pooled features (B, C)
        pooled = torch.mean(dropped, dim=1)

        # Compute gating vector via sigmoid applied to pooled features (B, C)
        gate = self.sigmoid(pooled)

        # Take the final time-step representation (B, C)
        last_step = dropped[:, -1, :]

        # Blend last-step and pooled via the gate: out = gate * last_step + (1 - gate) * pooled
        out = gate * last_step + (1.0 - gate) * pooled

        return out

# Configuration (module-level)
batch_size = 8
seq_len = 64
input_size = 128
hidden_size = 256
num_layers = 2
nonlinearity = "tanh"
dropout_p = 0.3
bidirectional = True

def get_inputs():
    """
    Returns:
        [x, h0] where:
          x: (batch_size, seq_len, input_size)
          h0: (num_layers * num_directions, batch_size, hidden_size)
    """
    num_directions = 2 if bidirectional else 1
    x = torch.randn(batch_size, seq_len, input_size)
    # Initialize a non-zero random hidden state to test initial hidden handling
    h0 = torch.randn(num_layers * num_directions, batch_size, hidden_size)
    return [x, h0]

def get_init_inputs():
    """
    Returns initialization parameters matching Model.__init__:
      [input_size, hidden_size, num_layers, nonlinearity, dropout_p, bidirectional]
    """
    return [input_size, hidden_size, num_layers, nonlinearity, dropout_p, bidirectional]