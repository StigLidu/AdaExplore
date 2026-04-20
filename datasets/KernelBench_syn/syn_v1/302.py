import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex model that combines spatial average pooling, linear patch projection,
    a stack of Transformer encoder layers, and a reconstruction head that upsamples
    the transformer output back to the original spatial resolution.

    Execution flow:
      1. Input image tensor (B, C, H, W) is reduced to non-overlapping patches using AvgPool2d
         with kernel and stride equal to patch_size, producing (B, C, H_p, W_p).
      2. Patch features are rearranged to a sequence (B, S, C) where S = H_p * W_p.
      3. A linear projection maps per-patch C-dim features to an embedding dimension (embed_dim).
      4. Positional embeddings are added and the sequence is fed through a TransformerEncoder
         (num_layers of nn.TransformerEncoderLayer).
      5. The transformer outputs are projected to out_channels and reshaped to (B, out_channels, H_p, W_p).
      6. Result is upsampled back to (B, out_channels, H, W) using bilinear interpolation.
    """
    def __init__(
        self,
        embed_dim: int,
        nhead: int,
        num_layers: int,
        patch_size: int,
        out_channels: int,
        in_channels: int,
        height: int,
        width: int,
        dropout: float = 0.1
    ):
        """
        Initializes the model.

        Args:
            embed_dim (int): Transformer embedding dimension.
            nhead (int): Number of attention heads.
            num_layers (int): Number of Transformer encoder layers.
            patch_size (int): Patch size used by AvgPool2d (non-overlapping patches).
            out_channels (int): Number of channels in the reconstructed output.
            in_channels (int): Number of channels in the input images.
            height (int): Height of the input image (must be divisible by patch_size).
            width (int): Width of the input image (must be divisible by patch_size).
            dropout (float): Dropout applied in Transformer layers.
        """
        super(Model, self).__init__()

        assert height % patch_size == 0 and width % patch_size == 0, \
            "height and width must be divisible by patch_size"

        self.embed_dim = embed_dim
        self.nhead = nhead
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.height = height
        self.width = width

        # Number of patches along spatial dimensions
        self.h_p = height // patch_size
        self.w_p = width // patch_size
        self.seq_len = self.h_p * self.w_p

        # 1) Spatial reduction into non-overlapping patches
        # Using AvgPool2d with kernel_size=patch_size and stride=patch_size
        self.pool = nn.AvgPool2d(kernel_size=patch_size, stride=patch_size)

        # 2) Linear projection from per-patch channel dimension to embedding dimension
        # We treat each pooled cell's C-dim as a token feature vector to be projected.
        self.patch_proj = nn.Linear(in_channels, embed_dim)

        # 3) Positional embeddings for sequence positions
        self.pos_embedding = nn.Parameter(torch.randn(self.seq_len, embed_dim))

        # 4) Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=False  # we'll provide (S, B, E) to the encoder
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 5) Normalization after transformer
        self.norm = nn.LayerNorm(embed_dim)

        # 6) Output projection back to spatial channels and upsampling head
        self.output_proj = nn.Linear(embed_dim, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Reconstructed tensor of shape (B, out_channels, H, W).
        """
        # Validate input shape
        B, C, H, W = x.shape
        assert C == self.in_channels and H == self.height and W == self.width, \
            f"Expected input shape (B, {self.in_channels}, {self.height}, {self.width}), got {x.shape}"

        # 1) Pool to non-overlapping patches -> (B, C, H_p, W_p)
        patches = self.pool(x)  # average per patch
        # 2) Rearrange to sequence: (B, H_p, W_p, C) -> (B, S, C)
        B, C, H_p, W_p = patches.shape
        seq = patches.permute(0, 2, 3, 1).reshape(B, self.seq_len, C)

        # 3) Project patch channels to embedding dimension: (B, S, E)
        seq = self.patch_proj(seq)

        # 4) Prepare for Transformer: (S, B, E)
        seq = seq.permute(1, 0, 2)

        # 5) Add positional embeddings (pos: S x E) broadcasted over batch
        seq = seq + self.pos_embedding.unsqueeze(1)  # (S, 1, E) -> (S, B, E) by broadcasting

        # 6) Transformer encoding: input (S, B, E) -> output (S, B, E)
        seq = self.transformer(seq)

        # 7) Back to (B, S, E)
        seq = seq.permute(1, 0, 2)

        # 8) Normalize per token
        seq = self.norm(seq)

        # 9) Output projection from embedding to image channels per patch: (B, S, out_channels)
        out_seq = self.output_proj(seq)

        # 10) Reshape to spatial grid: (B, H_p, W_p, out_channels) -> (B, out_channels, H_p, W_p)
        out = out_seq.reshape(B, self.h_p, self.w_p, self.out_channels).permute(0, 3, 1, 2)

        # 11) Upsample back to original resolution
        out_upsampled = F.interpolate(out, scale_factor=self.patch_size, mode='bilinear', align_corners=False)

        return out_upsampled

# Module-level configuration
batch_size = 8
in_channels = 3
height = 128
width = 128
patch_size = 8
embed_dim = 256
nhead = 8
num_layers = 4
out_channels = 3  # reconstruct RGB-like output

def get_inputs():
    """
    Returns a list containing a single input tensor matching the expected input shape.
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters for the Model in the order expected by its constructor.
    """
    return [embed_dim, nhead, num_layers, patch_size, out_channels, in_channels, height, width]