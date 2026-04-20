import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A compact transformer-like block that demonstrates a sequence of projections,
    multi-head self-attention, gated residual connection, and a two-layer feed-forward
    network with non-linearities. The block also applies a final projection and L2
    normalization.

    Structure:
    1. Linear projections to produce Q, K, V.
    2. Multihead self-attention (batch_first=True).
    3. Residual connection + LeakyReLU gating.
    4. Two-layer feed-forward network with ReLU activation.
    5. Output projection + residual + L2 normalization.
    """
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.0):
        """
        Initializes the transformer-like block.

        Args:
            embed_dim: Dimensionality of input embeddings.
            num_heads: Number of attention heads.
            ff_dim: Hidden dimensionality of the feed-forward network.
            dropout: Dropout probability (applied to attention output projection).
        """
        super(Model, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim

        # Projections for query, key, value
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Multi-head attention (expects inputs in (batch, seq, embed) when batch_first=True)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=dropout)

        # Output projection after attention
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Non-linearities used in the block
        self.leaky = nn.LeakyReLU(negative_slope=0.1)
        self.relu = nn.ReLU()

        # Feed-forward network
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the transformer-like block.

        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim).

        Returns:
            Tensor of shape (batch, seq_len, embed_dim) after attention, feed-forward,
            and normalization.
        """
        # Project inputs to queries, keys, values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Multi-head self-attention (returns attn output in same shape (batch, seq, embed_dim))
        attn_out, _attn_weights = self.attn(q, k, v)

        # Residual connection and gated non-linearity
        # gate combines original input and attention output before non-linearity
        gated = self.leaky(attn_out + x)

        # Feed-forward network with non-linearity
        ff = self.fc2(self.relu(self.fc1(gated)))

        # Second residual connection and final projection
        out = self.out_proj(ff + gated)

        # L2-normalize across embedding dimension for numerical stability and comparability
        norm = torch.norm(out, p=2, dim=-1, keepdim=True)
        out = out / (norm + 1e-6)

        return out

# Configuration / shapes
BATCH_SIZE = 4
SEQ_LEN = 128
EMBED_DIM = 512
NUM_HEADS = 8
FF_DIM = 2048

def get_inputs():
    """
    Returns a list containing the input tensor for the model:
    - x: Random input tensor of shape (BATCH_SIZE, SEQ_LEN, EMBED_DIM)
    """
    x = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for the Model constructor:
    [embed_dim, num_heads, ff_dim]
    """
    return [EMBED_DIM, NUM_HEADS, FF_DIM]