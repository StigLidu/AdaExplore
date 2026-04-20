import torch
import torch.nn as nn
from typing import List

class Model(nn.Module):
    """
    Sequence processing model that combines an LSTM, SyncBatchNorm over the
    feature (channel) dimension across time, Dropout1d (channel dropout across time),
    and a final linear projection. The model demonstrates non-trivial tensor
    reshapes and normalization semantics suitable for sequence modeling tasks.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout_prob: float,
        bidirectional: bool = False,
    ):
        """
        Initialize the model.

        Args:
            input_size (int): Number of input features per time-step.
            hidden_size (int): Hidden size of the LSTM per direction.
            num_layers (int): Number of stacked LSTM layers.
            dropout_prob (float): Dropout probability for Dropout1d.
            bidirectional (bool): If True, use a bidirectional LSTM.
        """
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.feature_channels = hidden_size * self.num_directions

        # LSTM for sequence encoding (batch_first to accept (B, T, F))
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # SyncBatchNorm across the feature channels (applied on (B, C, T) layout)
        self.syncbn = nn.SyncBatchNorm(self.feature_channels)

        # Dropout1d drops entire channels consistently across the time dimension
        self.dropout1d = nn.Dropout1d(p=dropout_prob)

        # Final projection from aggregated features back to input feature dimensionality
        self.fc = nn.Linear(self.feature_channels, input_size)

        # Small activation for output
        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, input_size) - a pooled and
                          projected representation of the input sequence.
        """
        # 1) Encode sequence with LSTM -> out: (B, T, hidden * directions)
        out, (h_n, c_n) = self.lstm(x)

        # 2) Prepare for channel-wise BN/Dropout: (B, T, C) -> (B, C, T)
        out = out.permute(0, 2, 1)

        # 3) Apply SyncBatchNorm over the channel dimension
        out = self.syncbn(out)

        # 4) Apply Dropout1d which zeros entire channels across time
        out = self.dropout1d(out)

        # 5) Return to (B, T, C) for temporal aggregation
        out = out.permute(0, 2, 1)

        # 6) Temporal pooling (mean over time) -> (B, C)
        pooled = out.mean(dim=1)

        # 7) Final projection and activation -> (B, input_size)
        projected = self.fc(pooled)
        projected = self.act(projected)

        return projected

# Configuration variables
batch_size = 12
seq_len = 64
input_size = 128
hidden_size = 96
num_layers = 2
dropout_prob = 0.25
bidirectional = True

def get_inputs() -> List[torch.Tensor]:
    """
    Create a random input batch for the model.

    Returns:
        List[torch.Tensor]: [x] where x has shape (batch_size, seq_len, input_size)
    """
    x = torch.randn(batch_size, seq_len, input_size)
    return [x]

def get_init_inputs() -> List:
    """
    Return the initialization parameters for the Model constructor.

    Returns:
        List: [input_size, hidden_size, num_layers, dropout_prob, bidirectional]
    """
    return [input_size, hidden_size, num_layers, dropout_prob, bidirectional]