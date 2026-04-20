import torch
import torch.nn as nn

# Configuration variables
batch_size = 16
C = 64           # input channels
D = 8            # depth
H = 32           # height
W = 32           # width

pad_size = 1     # replication padding applied on all sides
embed_dim = 128  # transformer embedding dimension (must be divisible by num_heads)
num_heads = 8
ff_dim = 512     # feedforward dimension for transformer
dropout_prob = 0.2

class Model(nn.Module):
    """
    Complex module that:
    - Pads a 5D volumetric input using replication padding (ReplicationPad3d).
    - Collapses the depth dimension via averaging to produce 4D feature maps.
    - Applies spatial channel dropout (Dropout2d).
    - Projects per-spatial-location channel vectors into a transformer embedding space.
    - Processes the flattened spatial tokens with a TransformerEncoderLayer.
    - Reconstructs per-location channel outputs and returns a 4D tensor.

    Input shape: (batch_size, C, D, H, W)
    Output shape: (batch_size, C, H_padded, W_padded)
    """
    def __init__(self):
        super(Model, self).__init__()
        # 3D replication padding: single int pads all sides by that amount
        self.pad3d = nn.ReplicationPad3d(pad_size)
        # Spatial dropout that drops entire channels for 2D feature maps
        self.dropout2d = nn.Dropout2d(p=dropout_prob)
        # Linear projection from channels -> transformer embedding
        self.proj = nn.Linear(C, embed_dim)
        # Transformer encoder layer operates on sequence of spatial tokens
        # Using default activation and dropout inside the transformer layer
        self.transformer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=False  # we will provide (seq_len, batch, embed_dim)
        )
        # Linear to reconstruct embeddings back to channel dimension
        self.reconstruct = nn.Linear(embed_dim, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, C, D, H, W)
        Returns:
            out: Tensor of shape (B, C, H_padded, W_padded)
        """
        # 1) Pad volumetric input
        # After padding: (B, C, D + 2*pad, H + 2*pad, W + 2*pad)
        x = self.pad3d(x)

        # 2) Collapse depth dimension by averaging, producing 4D feature maps
        # Result: (B, C, H_p, W_p)
        x = x.mean(dim=2)

        # 3) Apply spatial channel dropout (works on 4D tensors)
        x = self.dropout2d(x)

        # 4) Prepare sequence of spatial tokens:
        # permute to (B, H_p, W_p, C) -> flatten spatial dims -> (B, S, C)
        B, Cc, Hp, Wp = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, Hp * Wp, Cc)  # (B, S, C)

        # 5) Project channels -> embedding for transformer
        x = self.proj(x)  # (B, S, embed_dim)

        # 6) Transformer expects (S, B, E)
        x = x.permute(1, 0, 2)  # (S, B, E)
        x = self.transformer(x)  # (S, B, E)

        # 7) Back to (B, S, E), reconstruct to channels and reshape to (B, C, Hp, Wp)
        x = x.permute(1, 0, 2)  # (B, S, E)
        x = self.reconstruct(x)  # (B, S, C)
        x = x.reshape(B, Hp, Wp, Cc).permute(0, 3, 1, 2)  # (B, C, Hp, Wp)

        return x

def get_inputs():
    """
    Returns input tensors for the model:
    - A single volumetric tensor of shape (batch_size, C, D, H, W)
    """
    A = torch.randn(batch_size, C, D, H, W, dtype=torch.float32)
    return [A]

def get_init_inputs():
    """
    No special initialization inputs required for this module.
    """
    return []