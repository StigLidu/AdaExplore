import torch
import torch.nn as nn
from typing import Tuple, List

class Model(nn.Module):
    """
    Complex 3D-to-sequence module that:
      - Applies a 3D average pooling to reduce spatial resolution.
      - Projects pooled voxels into an embedding space to form a token sequence (tgt).
      - Creates a compact memory sequence from a global pooled descriptor.
      - Runs a TransformerDecoderLayer over the tgt with the memory.
      - Applies a Hardswish activation and projects back to channel space, returning a pooled-volume tensor.

    This combines nn.AvgPool3d, nn.TransformerDecoderLayer and nn.Hardswish in a compact processing pipeline.
    """
    def __init__(
        self,
        in_channels: int,
        pool_kernel: Tuple[int, int, int],
        embed_dim: int,
        nhead: int,
        mem_len: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        """
        Args:
            in_channels: Number of input channels in the 3D volume.
            pool_kernel: 3-tuple kernel size for AvgPool3d (also used as stride for simplicity).
            embed_dim: Embedding dimension for the transformer.
            nhead: Number of attention heads for the transformer decoder layer.
            mem_len: Length of the memory sequence produced from the global descriptor.
            dim_feedforward: Feedforward dimension inside TransformerDecoderLayer.
            dropout: Dropout for the TransformerDecoderLayer.
        """
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.pool_kernel = pool_kernel
        self.embed_dim = embed_dim
        self.mem_len = mem_len

        # 3D average pooling to reduce spatial resolution and complexity
        self.pool = nn.AvgPool3d(kernel_size=pool_kernel, stride=pool_kernel, padding=0)

        # Linear projections between channel space and transformer embedding space
        self.proj_tgt = nn.Linear(in_channels, embed_dim)   # projects each voxel/channel vector -> embedding
        self.proj_mem = nn.Linear(in_channels, embed_dim)   # projects global descriptor -> memory embedding
        self.proj_back = nn.Linear(embed_dim, in_channels)  # projects back to original channel dimension

        # Transformer decoder layer that will process the sequence with memory
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=False  # we will use (S, N, E) convention
        )

        # Non-linearity applied after transformer
        self.act = nn.Hardswish()

        # Small LayerNorm for stability on embeddings
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, D, H, W)

        Returns:
            Tensor of shape (B, C, D', H', W') where D',H',W' are reduced by pool_kernel.
        """
        if x.dim() != 5:
            raise ValueError("Expected input of shape (B, C, D, H, W)")

        B, C, D, H, W = x.shape

        # 1) Reduce spatial dimensions with AvgPool3d
        x_pooled = self.pool(x)  # shape: (B, C, D', H', W')
        Bp, Cp, Dp, Hp, Wp = x_pooled.shape  # Bp == B, Cp == C

        # 2) Form token sequence from voxels: flatten spatial dims -> tokens
        # src_seq: (seq_len, B, C)
        seq = Dp * Hp * Wp
        src_seq = x_pooled.flatten(2).permute(2, 0, 1)  # (seq, B, C)

        # 3) Project tokens into embedding space -> tgt (seq, B, E)
        tgt = self.proj_tgt(src_seq)  # (seq, B, embed_dim)
        tgt = self.norm(tgt)         # LayerNorm on embedding dims

        # 4) Create a compact memory sequence from global pooled descriptor:
        #    global descriptor per batch: mean across spatial dims -> (B, C)
        global_desc = x_pooled.mean(dim=(2, 3, 4))  # (B, C)
        # project to embedding and expand to mem_len
        mem_emb = self.proj_mem(global_desc).unsqueeze(0)  # (1, B, E)
        memory = mem_emb.repeat(self.mem_len, 1, 1)        # (mem_len, B, E)

        # 5) TransformerDecoderLayer: uses tgt and memory
        #    Output shape: (seq, B, E)
        decoded = self.decoder_layer(tgt, memory)

        # 6) Residual combine with original tgt embedding for stability
        decoded = decoded + tgt

        # 7) Activation and project back to channel dimension
        activated = self.act(decoded)                 # (seq, B, E)
        projected = self.proj_back(activated)        # (seq, B, C)

        # 8) Reshape back to (B, C, D', H', W')
        projected = projected.permute(1, 2, 0)  # (B, C, seq)
        out = projected.view(B, C, Dp, Hp, Wp)  # (B, C, D', H', W')

        return out


# Configuration variables (module-level)
batch_size = 4
in_channels = 8
D = 16
H = 16
W = 8

# Pool kernel: reduces the spatial dims; choose factors that divide the dims reasonably
pool_kernel = (4, 4, 2)   # results in D'=4, H'=4, W'=4 for the chosen D,H,W
embed_dim = 64
nhead = 8
mem_len = 2
dim_feedforward = 128
dropout = 0.05

def get_inputs() -> List[torch.Tensor]:
    """
    Returns:
        A single input tensor with shape (batch_size, in_channels, D, H, W)
    """
    x = torch.randn(batch_size, in_channels, D, H, W)
    return [x]

def get_init_inputs() -> List:
    """
    Returns:
        Initialization arguments for Model.__init__ in the expected order:
        (in_channels, pool_kernel, embed_dim, nhead, mem_len, dim_feedforward, dropout)
    """
    return [in_channels, pool_kernel, embed_dim, nhead, mem_len, dim_feedforward, dropout]