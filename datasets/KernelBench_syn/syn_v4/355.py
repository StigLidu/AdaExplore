import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex model combining 3D Batch Normalization (lazy), spatial pooling,
    an LSTM over temporal dimension, and a final classification with Softmax.

    Expected input shape: (batch_size, seq_len, channels, depth, height, width)
    The model normalizes each 3D frame with a shared LazyBatchNorm3d, reduces
    spatial dimensions via AdaptiveAvgPool3d to a per-channel descriptor,
    feeds the sequence of descriptors into an LSTM, then projects the final
    LSTM output through a linear layer and Softmax to produce class probabilities.
    """
    def __init__(
        self,
        input_channels: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_dim: int = 10,
        bidirectional: bool = False,
        dropout: float = 0.0
    ):
        """
        Initializes the module components.

        Args:
            input_channels (int): Number of channels in each 3D frame.
            hidden_size (int): Hidden size of the LSTM.
            num_layers (int): Number of stacked LSTM layers.
            output_dim (int): Number of output classes.
            bidirectional (bool): Whether to use a bidirectional LSTM.
            dropout (float): Dropout probability applied after the FC layer.
        """
        super(Model, self).__init__()
        # Lazy batch norm will infer `num_features` on first forward when it sees channels.
        self.bn3d = nn.LazyBatchNorm3d()
        # Pool each 3D volume to a single value per channel -> (N, C, 1, 1, 1)
        self.pool3d = nn.AdaptiveAvgPool3d((1, 1, 1))
        # LSTM expects input_size == input_channels (we will squeeze pooled dims)
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        lstm_out_size = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(lstm_out_size, output_dim)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        # Softmax over class dimension
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C, D, H, W).

        Returns:
            torch.Tensor: Class probabilities of shape (B, output_dim).
        """
        # Validate dimensions
        if x.ndim != 6:
            raise ValueError("Input must be 6D tensor (B, T, C, D, H, W)")

        B, T, C, D, H, W = x.shape

        # Merge batch and time to apply 3D batch norm across all frames
        x_merged = x.view(B * T, C, D, H, W)            # (B*T, C, D, H, W)
        x_norm = self.bn3d(x_merged)                    # (B*T, C, D, H, W)

        # Spatial pooling to reduce D,H,W -> 1,1,1 turning each frame into a channel vector
        x_pooled = self.pool3d(x_norm).view(B, T, C)    # (B, T, C)

        # Pass the per-timestep channel descriptors through LSTM
        lstm_out, _ = self.lstm(x_pooled)               # (B, T, hidden*directions)

        # Use the final timestep output for classification
        final_step = lstm_out[:, -1, :]                 # (B, hidden*directions)
        final_step = self.dropout(final_step)
        logits = self.fc(final_step)                    # (B, output_dim)

        # Convert logits to probabilities
        probs = self.softmax(logits)                    # (B, output_dim)
        return probs

# Configuration / default parameters
batch_size = 8
seq_len = 6
channels = 12
depth = 8
height = 16
width = 16
hidden_size = 64
num_layers = 2
output_dim = 5
bidirectional = True
dropout = 0.1

def get_inputs():
    """
    Returns a list containing the input tensor shaped (B, T, C, D, H, W).
    """
    x = torch.randn(batch_size, seq_len, channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for Model __init__:
    [input_channels, hidden_size, num_layers, output_dim, bidirectional, dropout]
    """
    return [channels, hidden_size, num_layers, output_dim, bidirectional, dropout]