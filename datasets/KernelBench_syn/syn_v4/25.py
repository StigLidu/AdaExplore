import torch
import torch.nn as nn

# Configuration variables
BATCH = 8
IN_CHANNELS = 64
HEIGHT = 32
WIDTH = 32
EMBED_DIM = 128
NUM_HEADS = 8
DROPOUT = 0.1  # used inside attention (if desired)

class Model(nn.Module):
    """
    A compact vision-attention block that:
      - Projects per-pixel channel features to an embedding space via a 1x1 conv.
      - Treats spatial positions as a sequence and applies MultiheadAttention across positions.
      - Uses a residual connection + LayerNorm over the embedding dimension.
      - Projects back to the original channel dimension and applies Softmax2d to produce
        a per-pixel probability distribution over channels.

    Input:
        x: Tensor of shape (batch, in_channels, H, W)

    Output:
        out: Tensor of shape (batch, in_channels, H, W) where channels at each spatial
             location sum to 1 (softmax over channel dimension).
    """
    def __init__(self,
                 in_channels: int = IN_CHANNELS,
                 embed_dim: int = EMBED_DIM,
                 num_heads: int = NUM_HEADS,
                 dropout: float = DROPOUT):
        super(Model, self).__init__()

        # Project input channels to embedding dimension (per spatial location)
        self.input_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=True)

        # Multi-head attention over spatial locations (sequence length = H*W).
        # The default MultiheadAttention interface expects inputs of shape (L, N, E).
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)

        # Normalize across embedding features after the residual add
        self.ln = nn.LayerNorm(embed_dim)

        # Project embedding back to original channel dimension
        self.output_proj = nn.Conv2d(embed_dim, in_channels, kernel_size=1, bias=True)

        # Softmax over channels for each spatial location
        self.softmax2d = nn.Softmax2d()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the attention-then-softmax block.

        Steps:
          1. input_proj: (B, C, H, W) -> (B, E, H, W)
          2. Flatten spatial dims: -> (L, B, E) where L = H*W
          3. Apply multi-head self-attention across spatial positions:
             attn_out, _ = attn(seq, seq, seq)
          4. Residual add and LayerNorm over the embedding dim
          5. Reshape back to (B, E, H, W)
          6. output_proj -> (B, C, H, W)
          7. Softmax2d to get per-pixel channel distributions

        Args:
            x: Tensor of shape (B, C, H, W)

        Returns:
            Tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        # 1. Project to embedding
        emb = self.input_proj(x)  # (B, E, H, W)

        # 2. Flatten spatial dims and make sequence (L, B, E)
        L = H * W
        seq = emb.flatten(2).permute(2, 0, 1)  # (L, B, E)

        # 3. Self-attention over spatial positions
        attn_out, _ = self.attn(seq, seq, seq)  # (L, B, E)

        # 4. Residual + LayerNorm
        res = attn_out + seq  # (L, B, E)
        # LayerNorm normalizes last dim (E)
        normed = self.ln(res)  # (L, B, E)

        # 5. Reshape back to (B, E, H, W)
        normed = normed.permute(1, 2, 0).contiguous().view(B, -1, H, W)  # (B, E, H, W)

        # 6. Project back to original channels
        out_proj = self.output_proj(normed)  # (B, C, H, W)

        # 7. Softmax2d over channels per spatial location
        out = self.softmax2d(out_proj)  # (B, C, H, W)

        return out

def get_inputs():
    """
    Returns typical input tensors for running the module.
    """
    x = torch.randn(BATCH, IN_CHANNELS, HEIGHT, WIDTH)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters used to construct the Model.
    """
    return [IN_CHANNELS, EMBED_DIM, NUM_HEADS, DROPOUT]