import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Volumetric encoder-decoder style module that:
    - Projects a 3D volume into an embedding via Conv3d
    - Downsamples with MaxPool3d (saving indices)
    - Treats the pooled spatial locations as a sequence and processes them with a single nn.TransformerDecoderLayer
      using a small learned set of memory tokens constructed from a linear projection of pooled features.
    - Reconstructs spatial volume via MaxUnpool3d using the saved indices and a skip connection
    - Applies channel-wise spatial dropout (Dropout2d) by collapsing one spatial dimension to fit its 4D requirement
    - Final 1x1 Conv3d to produce the output channels
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embed_dim: int,
        pool_kernel: int,
        nhead: int,
        mem_len: int,
        dropout_p: float = 0.1,
    ):
        super(Model, self).__init__()
        assert embed_dim % nhead == 0, "embed_dim must be divisible by nhead"

        # Initial projection into embedding space
        self.conv_in = nn.Conv3d(in_channels, embed_dim, kernel_size=3, padding=1)

        # Downsample and corresponding unpool
        self.pool = nn.MaxPool3d(kernel_size=pool_kernel, stride=pool_kernel, return_indices=True)
        self.unpool = nn.MaxUnpool3d(kernel_size=pool_kernel, stride=pool_kernel)

        # Transformer decoder layer: operates on sequences of length (D'*H'*W')
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=dropout_p,
            activation='relu'
        )

        # Small projection to create mem_len memory tokens per batch
        self.mem_len = mem_len
        self.memory_proj = nn.Linear(embed_dim, mem_len * embed_dim)

        # Dropout2d expects 4D input (N, C, H, W). We'll collapse one spatial dim to fit.
        self.dropout2d = nn.Dropout2d(p=dropout_p)

        # Final projection back to desired output channels
        self.final_conv = nn.Conv3d(embed_dim, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input volumetric tensor of shape (B, C_in, D, H, W)

        Returns:
            Tensor of shape (B, C_out, D, H, W)
        """
        # Record original spatial shape for unpooling
        orig_shape = x.shape  # (B, C_in, D, H, W)

        # 1) Project to embedding
        x_emb = self.conv_in(x)  # (B, E, D, H, W)
        x_emb = F.relu(x_emb)

        # 2) MaxPool3d downsample and save indices
        pooled, indices = self.pool(x_emb)  # pooled: (B, E, Dp, Hp, Wp)

        B, E, Dp, Hp, Wp = pooled.shape

        # 3) Create sequence from pooled spatial locations: seq_len = Dp * Hp * Wp
        seq = pooled.view(B, E, -1).permute(2, 0, 1)  # (seq_len, B, E)

        # 4) Build memory tokens: project pooled global representation into mem_len tokens
        #    pooled_mean: (B, E)
        pooled_mean = seq.mean(dim=0)  # (B, E)
        mem = self.memory_proj(pooled_mean)  # (B, mem_len * E)
        mem = mem.view(B, self.mem_len, E).permute(1, 0, 2)  # (mem_len, B, E)

        # 5) Transformer decoding: use seq as target, mem as memory
        decoded = self.decoder_layer(seq, mem)  # (seq_len, B, E)

        # 6) Restore spatial layout
        decoded_vol = decoded.permute(1, 2, 0).view(B, E, Dp, Hp, Wp)  # (B, E, Dp, Hp, Wp)

        # 7) Unpool to original embedding spatial resolution
        #    Provide output_size as the embedding tensor shape before pooling
        unpooled = self.unpool(decoded_vol, indices, output_size=x_emb.shape)  # (B, E, D, H, W)

        # 8) Skip connection with the pre-pooled embedding
        combined = x_emb + unpooled  # (B, E, D, H, W)

        # 9) Apply Dropout2d: collapse D and H into one dimension to make a 4D tensor
        b, c, d, h, w = combined.shape
        collapsed = combined.view(b, c, d * h, w)  # (B, E, D*H, W)
        dropped = self.dropout2d(collapsed)
        combined = dropped.view(b, c, d, h, w)  # (B, E, D, H, W)

        # 10) Final 1x1x1 conv to produce outputs
        out = self.final_conv(combined)  # (B, out_channels, D, H, W)
        return out

# Module-level configuration
BATCH_SIZE = 2
IN_CHANNELS = 3
DEPTH = 16
HEIGHT = 32
WIDTH = 32
OUT_CHANNELS = 2
EMBED_DIM = 64  # must be divisible by NHEAD
POOL_KERNEL = 2
NHEAD = 8
MEM_LEN = 4
DROPOUT_P = 0.15

def get_inputs():
    """
    Returns a list with a single input tensor representing a small 3D volume batch.
    Shape: (BATCH_SIZE, IN_CHANNELS, DEPTH, HEIGHT, WIDTH)
    """
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, DEPTH, HEIGHT, WIDTH)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters for the Model constructor in the same order.
    """
    return [IN_CHANNELS, OUT_CHANNELS, EMBED_DIM, POOL_KERNEL, NHEAD, MEM_LEN, DROPOUT_P]