import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex model that:
    - Normalizes image input with BatchNorm2d
    - Uses a Conv2d patch embedding to convert image into tokens
    - Adds learned positional encodings
    - Processes tokens with a TransformerEncoder stack
    - Pools transformer outputs, applies LazyBatchNorm1d and an MLP gating
    - Modulates transformer tokens with the pooled gating and reconstructs the image via ConvTranspose2d

    The model returns a reconstructed image tensor of the same spatial size as the input.
    """
    def __init__(
        self,
        in_channels: int,
        image_size: int,
        patch_size: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
    ):
        """
        Initializes the model components.

        Args:
            in_channels (int): Number of channels in the input images (e.g., 3 for RGB).
            image_size (int): Input image spatial size (assumes square images H = W = image_size).
            patch_size (int): Size of patches to extract (must divide image_size).
            embed_dim (int): Token embedding dimensionality.
            num_layers (int): Number of TransformerEncoder layers.
            num_heads (int): Number of attention heads in each Transformer layer.
            ff_dim (int): Hidden dimension for Transformer feedforward networks.
        """
        super(Model, self).__init__()

        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.in_channels = in_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Normalization over 4D image input
        self.bn2d = nn.BatchNorm2d(in_channels)

        # Patch embedding: conv with kernel and stride equal to patch_size produces non-overlapping patches
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)

        # Number of patches (sequence length)
        self.h_p = image_size // patch_size
        self.w_p = image_size // patch_size
        self.num_patches = self.h_p * self.w_p

        # Learned positional encodings for the patch sequence
        self.positional_encoding = nn.Parameter(torch.randn(self.num_patches, embed_dim))

        # Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, activation='gelu', batch_first=False)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Pooling normalization (lazy so we don't need to know embed_dim for serialization)
        self.pool_norm = nn.LazyBatchNorm1d()

        # Small MLP to create gating vector from pooled representation
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Final deconvolution to reconstruct image from token map
        self.deconv = nn.ConvTranspose2d(embed_dim, in_channels, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, in_channels, image_size, image_size)

        Returns:
            torch.Tensor: Reconstructed image tensor of same shape as input.
        """
        # Normalize input
        x = self.bn2d(x)

        # Patch embedding -> (B, E, Hp, Wp)
        patches = self.patch_embed(x)
        B, E, Hp, Wp = patches.shape  # Hp == self.h_p, Wp == self.w_p

        # Flatten spatial patches into sequence -> (seq_len, B, E) for TransformerEncoder
        seq = patches.view(B, E, Hp * Wp).permute(2, 0, 1)  # (S, B, E), S = num_patches

        # Add positional encodings (pos: (S, E) -> (S, B, E) via broadcasting)
        seq = seq + self.positional_encoding.unsqueeze(1)

        # Transformer expects (S, B, E)
        seq = self.transformer(seq)  # (S, B, E)

        # Bring to (B, S, E)
        seq_b = seq.permute(1, 0, 2)

        # Pool across sequence to get global descriptor
        pooled = seq_b.mean(dim=1)  # (B, E)

        # Normalize pooled features lazily
        pooled = self.pool_norm(pooled)

        # Compute gating vector and apply sigmoid
        gate = torch.sigmoid(self.mlp_head(pooled))  # (B, E)

        # Modulate token sequence by gate
        gated_seq = seq_b * gate.unsqueeze(1)  # (B, S, E)

        # Reshape back to (B, E, Hp, Wp)
        token_map = gated_seq.permute(0, 2, 1).view(B, E, Hp, Wp)

        # Reconstruct image via transposed convolution
        recon = self.deconv(token_map)  # (B, in_channels, image_size, image_size)

        return recon

# Configuration variables
batch_size = 8
in_channels = 3
image_size = 64
patch_size = 8
embed_dim = 128
num_layers = 3
num_heads = 4
ff_dim = 256

def get_inputs():
    """
    Creates a batch of random images.

    Returns:
        list: single-element list containing the input tensor.
    """
    x = torch.randn(batch_size, in_channels, image_size, image_size)
    return [x]

def get_init_inputs():
    """
    Returns initialization inputs necessary to construct the Model.

    Returns:
        list: [in_channels, image_size, patch_size, embed_dim, num_layers, num_heads, ff_dim]
    """
    return [in_channels, image_size, patch_size, embed_dim, num_layers, num_heads, ff_dim]