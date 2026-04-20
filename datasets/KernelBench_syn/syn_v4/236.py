import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Patch-based Transformer module that:
    - Extracts non-overlapping patches from an image using nn.Unfold
    - Linearly projects patches to a d_model dimensional embedding
    - Adds learned positional embeddings
    - Processes the sequence of patch tokens with an nn.TransformerEncoder stack
    - Applies nn.PReLU non-linearity across the embedding channels
    - Decodes tokens back to patch pixels and reconstructs the image with nn.Fold
    - Adds a residual connection from the input image to the reconstructed output

    This creates a hybrid convolution/transformer pattern suitable for vision-style inputs.
    """
    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        height: int,
        width: int,
    ):
        super(Model, self).__init__()

        assert height % patch_size == 0 and width % patch_size == 0, "Height and width must be divisible by patch_size"

        self.in_channels = in_channels
        self.patch_size = patch_size
        self.d_model = d_model
        self.height = height
        self.width = width

        # Unfold / Fold for patch extraction and reconstruction (non-overlapping patches)
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.fold = nn.Fold(output_size=(height, width), kernel_size=patch_size, stride=patch_size)

        # Number of patches (sequence length)
        self.num_patches_h = height // patch_size
        self.num_patches_w = width // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

        # Dimension of a raw patch vector (C * patch_size * patch_size)
        self.patch_dim = in_channels * patch_size * patch_size

        # Linear projection from patch vector -> transformer embedding
        self.proj = nn.Linear(self.patch_dim, d_model)

        # Positional embeddings for each patch (learned)
        self.pos_embed = nn.Parameter(torch.randn(self.num_patches, d_model))

        # Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Non-linearity applied across embedding channels (we will permute dims so channels dim is correct)
        self.prelu = nn.PReLU(num_parameters=d_model)

        # Decoder: map transformer embeddings back to raw patch vectors
        self.decoder = nn.Linear(d_model, self.patch_dim)

        # Small initialization tweaks (optional)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.decoder.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image tensor of shape (batch_size, in_channels, height, width)

        Returns:
            Reconstructed tensor of same shape (batch_size, in_channels, height, width)
        """
        # Extract patches -> shape (B, patch_dim, L)
        patches = self.unfold(x)  # B, patch_dim, L
        # Move patches to (B, L, patch_dim)
        patches = patches.transpose(1, 2)  # B, L, patch_dim

        # Linear projection to embeddings (B, L, d_model)
        tokens = self.proj(patches)

        # Transformer expects (S, N, E) = (seq_len, batch_size, embed_dim)
        tokens = tokens.transpose(0, 1)  # L, B, d_model

        # Add positional embeddings (L, 1, d_model) -> broadcast to (L, B, d_model)
        tokens = tokens + self.pos_embed.unsqueeze(1)

        # Transformer encoding (L, B, d_model)
        encoded = self.transformer(tokens)

        # Back to (B, L, d_model)
        encoded = encoded.transpose(0, 1)

        # Apply PReLU across embedding channels: permute to (B, d_model, L) so channel dim = 1
        encoded = encoded.permute(0, 2, 1)  # B, d_model, L
        activated = self.prelu(encoded)
        activated = activated.permute(0, 2, 1)  # B, L, d_model

        # Decode back to patch vectors (B, L, patch_dim)
        recon_patches = self.decoder(activated)

        # Move to (B, patch_dim, L) for folding
        recon_patches = recon_patches.transpose(1, 2)  # B, patch_dim, L

        # Reconstruct image
        out = self.fold(recon_patches)  # B, C, H, W

        # Residual connection from input -> helps training stability
        # Ensure shapes match and add
        return out + x


# Configuration / default parameters
batch_size = 8
in_channels = 3
height = 64
width = 64
patch_size = 8
d_model = 128
nhead = 8
num_layers = 2
dim_feedforward = 256
dropout = 0.1

def get_inputs():
    """
    Returns:
        A list containing a single input image tensor with shape (batch_size, in_channels, height, width).
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns:
        Initialization arguments for Model.__init__ in the same order as the signature.
    """
    return [in_channels, patch_size, d_model, nhead, num_layers, dim_feedforward, dropout, height, width]