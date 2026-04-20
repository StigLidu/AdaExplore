import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Image-to-Image Transformer block that:
    - Uses PixelUnshuffle to split spatial resolution into channel patches
    - Projects patches into a transformer-friendly embedding space
    - Processes patches with a stack of TransformerEncoder layers
    - Applies AlphaDropout for regularization in embedding space
    - Projects back to patch channels and reconstructs the high-resolution image

    This module demonstrates a non-trivial combination of PixelUnshuffle, TransformerEncoder,
    and AlphaDropout to form an image processing pipeline.
    """
    def __init__(
        self,
        in_channels: int,
        height: int,
        width: int,
        downscale_factor: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float,
        alpha_dropout_p: float,
    ):
        """
        Args:
            in_channels: number of input image channels (C)
            height: image height (H), must be divisible by downscale_factor
            width: image width (W), must be divisible by downscale_factor
            downscale_factor: pixel unshuffle factor r (must divide H and W)
            d_model: embedding dimension for the transformer
            nhead: number of attention heads
            num_layers: number of TransformerEncoder layers
            dropout: dropout probability inside TransformerEncoderLayer
            alpha_dropout_p: dropout probability for AlphaDropout applied after encoder
        """
        super(Model, self).__init__()

        assert height % downscale_factor == 0 and width % downscale_factor == 0, \
            "height and width must be divisible by downscale_factor"

        self.in_channels = in_channels
        self.H = height
        self.W = width
        self.r = downscale_factor

        # PixelUnshuffle reduces spatial dims by r and increases channels by r^2
        self.unshuffle = nn.PixelUnshuffle(self.r)

        # Number of channels per spatial patch after unshuffle
        self.patch_channels = in_channels * (self.r ** 2)
        # Sequence length (number of patches)
        self.Hr = height // self.r
        self.Wr = width // self.r
        self.seq_len = self.Hr * self.Wr

        # Linear projection from patch_channels -> d_model (applied per patch)
        self.patch_proj = nn.Linear(self.patch_channels, d_model)

        # Positional embeddings for patches (seq_len, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(self.seq_len, d_model))

        # Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # AlphaDropout for preserving self-normalizing activations statistics
        self.alpha_dropout = nn.AlphaDropout(p=alpha_dropout_p)

        # Optional normalization and final projection back to patch channels
        self.layer_norm = nn.LayerNorm(d_model)
        self.reconstruct_proj = nn.Linear(d_model, self.patch_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input image tensor of shape (B, C, H, W)

        Returns:
            Reconstructed image tensor of same shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        assert C == self.in_channels and H == self.H and W == self.W, \
            f"Expected input shape (B, {self.in_channels}, {self.H}, {self.W}), got {x.shape}"

        # 1) PixelUnshuffle: (B, C, H, W) -> (B, C*r*r, H/r, W/r)
        x_un = self.unshuffle(x)  # (B, patch_channels, Hr, Wr)

        # 2) Flatten spatial dims to sequence of patches:
        #    (B, patch_channels, Hr, Wr) -> (B, seq_len, patch_channels)
        B_, PC, Hr, Wr = x_un.shape
        seq_len = Hr * Wr
        x_un_flat = x_un.view(B_, PC, seq_len).permute(0, 2, 1)  # (B, seq_len, patch_channels)

        # 3) Linear projection to embeddings: (B, seq_len, d_model)
        emb = self.patch_proj(x_un_flat)

        # 4) Prepare for transformer: transformer expects (S, N, E)
        emb_t = emb.permute(1, 0, 2)  # (seq_len, B, d_model)

        # 5) Add positional embeddings (seq_len, 1, d_model) broadcast to (seq_len, B, d_model)
        emb_t = emb_t + self.pos_embedding.unsqueeze(1)

        # 6) Transformer encoder stack
        encoded = self.transformer(emb_t)  # (seq_len, B, d_model)

        # 7) Back to batch-first (B, seq_len, d_model)
        encoded_b = encoded.permute(1, 0, 2)

        # 8) LayerNorm + AlphaDropout
        encoded_norm = self.layer_norm(encoded_b)
        encoded_drop = self.alpha_dropout(encoded_norm)

        # 9) Project back to patch channels and reshape to spatial grid:
        #    (B, seq_len, patch_channels) -> (B, patch_channels, Hr, Wr)
        recon_patches = self.reconstruct_proj(encoded_drop)  # (B, seq_len, patch_channels)
        recon_patches = recon_patches.permute(0, 2, 1).contiguous().view(B, self.patch_channels, Hr, Wr)

        # 10) Reconstruct high-resolution image by reversing PixelUnshuffle:
        #     recon_patches.view(B, C, r, r, Hr, Wr) -> permute/reshape -> (B, C, H, W)
        recon = recon_patches.view(B, C, self.r, self.r, Hr, Wr)
        recon = recon.permute(0, 1, 4, 2, 5, 3).contiguous().view(B, C, H, W)

        return recon

# Configuration variables
batch_size = 8
in_channels = 3
height = 128
width = 128
downscale_factor = 4  # r
d_model = 256
nhead = 8
num_layers = 4
dropout = 0.1
alpha_dropout_p = 0.05

def get_inputs():
    """
    Returns example input tensors for running the Model.forward.
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization arguments for constructing the Model.
    Order matches Model.__init__ signature.
    """
    return [in_channels, height, width, downscale_factor, d_model, nhead, num_layers, dropout, alpha_dropout_p]