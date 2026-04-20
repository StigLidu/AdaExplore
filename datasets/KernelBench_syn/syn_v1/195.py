import torch
import torch.nn as nn
from typing import List

class Model(nn.Module):
    """
    Sequence processing model that projects inputs, applies an LSTM, and performs
    gated non-linear transformations with residual connections and clipping.

    The computation graph is:
      1. input projection: x -> embed (Linear)
      2. LSTM over sequence -> lstm_out
      3. linear projection of LSTM outputs -> gated (Linear)
      4. non-linearity: Softplus applied to gated output
      5. residual add with original embed
      6. clipping non-linearity: Hardtanh
      7. final projection back to input feature size

    This model demonstrates combining nn.LSTM, nn.Softplus, and nn.Hardtanh
    in a multi-step sequence processing pipeline.
    """
    def __init__(
        self,
        input_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0
    ):
        """
        Initializes the model components.

        Args:
            input_size (int): Dimensionality of input features.
            embed_size (int): Size of intermediate embeddings.
            hidden_size (int): Hidden size for the LSTM.
            num_layers (int, optional): Number of LSTM layers. Defaults to 1.
            bidirectional (bool, optional): Whether LSTM is bidirectional. Defaults to False.
            dropout (float, optional): Dropout between LSTM layers. Defaults to 0.0.
        """
        super(Model, self).__init__()
        self.input_size = input_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1

        # Project input features to an embedding space before the LSTM
        self.input_proj = nn.Linear(input_size, embed_size)

        # LSTM processes the sequence; use batch_first for (B, T, C) tensors
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Project LSTM hidden outputs back to embedding size for gating/residual
        self.lstm_to_embed = nn.Linear(hidden_size * self.num_directions, embed_size)

        # Non-linearities
        self.softplus = nn.Softplus()
        # Hardtanh used to clip values to a stable range after residual add
        self.hardtanh = nn.Hardtanh(min_val=-6.0, max_val=6.0)

        # Final projection: returns features with original input_size
        self.output_proj = nn.Linear(embed_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, input_size).
        """
        # 1) Project input features to embedding space
        embed = self.input_proj(x)  # (B, T, embed_size)

        # 2) Run LSTM on embeddings
        lstm_out, _ = self.lstm(embed)  # (B, T, hidden_size * num_directions)

        # 3) Project LSTM outputs back to embedding size (prepare gating)
        gated = self.lstm_to_embed(lstm_out)  # (B, T, embed_size)

        # 4) Apply Softplus as a smooth gating non-linearity
        gated = self.softplus(gated)

        # 5) Residual connection: combine gated information with original embed
        combined = embed + gated  # (B, T, embed_size)

        # 6) Clip values to a bounded range for numerical stability
        clipped = self.hardtanh(combined)

        # 7) Final projection to reconstruct original feature dimensionality
        out = self.output_proj(clipped)  # (B, T, input_size)

        return out

# Configuration for input generation and model initialization
batch_size = 8
seq_len = 128
input_size = 64
embed_size = 128
hidden_size = 96
num_layers = 2
bidirectional = True
dropout = 0.1

def get_inputs() -> List[torch.Tensor]:
    """
    Returns synthetic input tensors matching the expected input shape:
      (batch_size, seq_len, input_size)

    The values are drawn from a standard normal distribution.
    """
    x = torch.randn(batch_size, seq_len, input_size)
    return [x]

def get_init_inputs() -> List:
    """
    Returns initialization parameters for the Model constructor, in order.
    """
    return [input_size, embed_size, hidden_size, num_layers, bidirectional, dropout]