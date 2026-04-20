import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A composite model that projects input token embeddings, adds learnable positional embeddings,
    processes the sequence with a stack of Transformer encoder layers, applies a Threshold non-linearity,
    and finally pools and projects the sequence output to produce per-batch representations.
    """
    def __init__(
        self,
        embed_dim: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        seq_len: int,
        threshold: float,
        dropout: float = 0.1,
    ):
        """
        Initializes the model components.

        Args:
            embed_dim (int): Dimensionality of token embeddings.
            nhead (int): Number of attention heads in the Transformer encoder.
            num_layers (int): Number of TransformerEncoderLayer layers to stack.
            dim_feedforward (int): Dimension of the feedforward network inside each Transformer layer.
            seq_len (int): Expected sequence length (used for positional embeddings).
            threshold (float): Threshold value for nn.Threshold layer.
            dropout (float): Dropout applied inside TransformerEncoderLayer.
        """
        super(Model, self).__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len

        # Optional input projection (identity if input already embed_dim, kept for flexibility)
        self.input_proj = nn.Linear(embed_dim, embed_dim)

        # Learnable positional embeddings for the fixed sequence length
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim))

        # Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=False,  # we'll feed (S, N, E)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Non-linearity after the encoder: Threshold will set values <= threshold to 0.0
        self.threshold = nn.Threshold(threshold, 0.0)

        # Final projection after pooling
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Small layer norm for stability
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, embed_dim)
        """
        # Project inputs (keeps shape)
        x = self.input_proj(x)  # (B, S, E)

        # Add positional embeddings (broadcast over batch)
        x = x + self.pos_embed  # (B, S, E)

        # Transformer expects (S, B, E) by default for batch_first=False
        x = x.permute(1, 0, 2)  # (S, B, E)

        # Pass through the Transformer encoder stack
        x = self.encoder(x)  # (S, B, E)

        # Apply threshold non-linearity elementwise
        x = self.threshold(x)  # (S, B, E)

        # Pool across sequence dimension (mean pooling)
        x = x.mean(dim=0)  # (B, E)

        # Optional normalization and final projection
        x = self.norm(x)
        x = self.out_proj(x)  # (B, E)

        return x

# Configuration variables
batch_size = 8
seq_len = 128
embed_dim = 512
nhead = 8
num_layers = 4
dim_feedforward = 2048
threshold_value = 0.05
dropout = 0.1

def get_inputs():
    """
    Returns:
        List containing a single input tensor of shape (batch_size, seq_len, embed_dim)
    """
    x = torch.randn(batch_size, seq_len, embed_dim)
    return [x]

def get_init_inputs():
    """
    Returns:
        Initialization parameters for the Model.__init__ in the following order:
        [embed_dim, nhead, num_layers, dim_feedforward, seq_len, threshold_value, dropout]
    """
    return [embed_dim, nhead, num_layers, dim_feedforward, seq_len, threshold_value, dropout]