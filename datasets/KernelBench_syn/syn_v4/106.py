import torch
import torch.nn as nn

"""
Complex PyTorch kernel combining ReflectionPad3d, TransformerEncoderLayer, and GELU.
- Pads a 3D volumetric input with reflection padding
- Projects channel features into a token embedding space
- Runs a single TransformerEncoderLayer across the flattened spatio-temporal tokens
- Applies GELU non-linearity and projects back to the original channel dimension
- Removes the padding and returns a tensor with the original spatial dimensions
"""

# Configuration / shapes
BATCH_SIZE = 8
CHANNELS = 3        # input channel dimension
DEPTH = 16          # D
HEIGHT = 32         # H
WIDTH = 32          # W

PAD = 1             # ReflectionPad3d padding on each side
EMBED_DIM = 64      # transformer embedding dimension (d_model)
NUM_HEADS = 8       # number of attention heads
FF_DIM = 256        # transformer feedforward hidden size

class Model(nn.Module):
    """
    Model that:
    - Pads volumetric input (B, C, D, H, W) using ReflectionPad3d
    - Flattens spatial dims into a token sequence (S, B, E) where E = EMBED_DIM
    - Applies a TransformerEncoderLayer (self-attention + feedforward)
    - Uses GELU and residual connections
    - Projects back to channel space and removes padding
    """
    def __init__(self,
                 embed_dim: int = EMBED_DIM,
                 num_heads: int = NUM_HEADS,
                 ff_dim: int = FF_DIM,
                 pad: int = PAD):
        super(Model, self).__init__()
        self.pad = nn.ReflectionPad3d(pad)
        # project input channel features -> token embedding
        self.proj_in = nn.Linear(CHANNELS, embed_dim)
        # small normalization before attention
        self.norm = nn.LayerNorm(embed_dim)
        # one transformer encoder layer (uses self-attn + feedforward)
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            activation='gelu',  # transformer internal activation
            batch_first=False   # transformer expects (S, N, E)
        )
        # explicit GELU activation included in the pipeline
        self.gelu = nn.GELU()
        # project back from embedding to channels
        self.proj_out = nn.Linear(embed_dim, CHANNELS)
        self._pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (B, C, D, H, W)

        Returns:
            out: output tensor of shape (B, C, D, H, W) (same spatial dims as input)
        """
        # 1) Reflection pad spatial dims (D, H, W)
        x_p = self.pad(x)  # (B, C, D+2p, H+2p, W+2p)

        # 2) Rearrange to token sequence: (S, B, C)
        B, C, Dp, Hp, Wp = x_p.shape
        # permute to (Dp, Hp, Wp, B, C) then flatten leading spatial dims -> (S, B, C)
        seq = x_p.permute(2, 3, 4, 0, 1).reshape(Dp * Hp * Wp, B, C)

        # 3) Project channels -> embed_dim and normalize
        seq = self.proj_in(seq)        # (S, B, E)
        seq = self.norm(seq)

        # 4) Transformer encoder layer (self-attention + feedforward)
        #    transformer layer operates on (S, B, E)
        t_out = self.transformer_layer(seq)  # (S, B, E)

        # 5) Residual connection + GELU nonlinearity
        seq = seq + t_out
        seq = self.gelu(seq)

        # 6) Project back to channel space
        seq = self.proj_out(seq)  # (S, B, C)

        # 7) Reshape tokens back to volumetric tensor (B, C, Dp, Hp, Wp)
        out_p = seq.reshape(Dp, Hp, Wp, B, C).permute(3, 4, 0, 1, 2)

        # 8) Crop to original spatial size by removing padding
        p = self._pad
        D, H, W = DEPTH, HEIGHT, WIDTH
        out = out_p[:, :, p:p + D, p:p + H, p:p + W]

        return out

# Provide initialization and input helpers to match the example structure
def get_inputs():
    """
    Returns:
        list with the input tensor(s) expected by Model.forward
    """
    x = torch.randn(BATCH_SIZE, CHANNELS, DEPTH, HEIGHT, WIDTH)
    return [x]

def get_init_inputs():
    """
    Returns:
        list of initialization parameters for the Model constructor in order:
        [embed_dim, num_heads, ff_dim, pad]
    """
    return [EMBED_DIM, NUM_HEADS, FF_DIM, PAD]