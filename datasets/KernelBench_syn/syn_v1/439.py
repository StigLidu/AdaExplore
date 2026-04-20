import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex model that:
    - Applies PixelUnshuffle to extract sub-pixel channel-expanded features.
    - Uses nn.Unfold to create a sequence of per-spatial-location tokens.
    - Projects tokens into a Transformer embedding space and processes them with nn.TransformerEncoder.
    - Projects tokens back, folds them to spatial layout, adds a residual connection, and uses pixel shuffle to reconstruct the original spatial resolution.

    This combines PixelUnshuffle, TransformerEncoder, and Fold (via functional.fold) into a coherent patch-transformer pipeline.
    """
    def __init__(
        self,
        in_channels: int,
        downscale_factor: int,
        d_model: int,
        num_layers: int,
        nhead: int,
        height: int,
        width: int,
        dropout: float = 0.1,
    ):
        """
        Initializes the model.

        Args:
            in_channels (int): Number of channels in input images.
            downscale_factor (int): Factor for PixelUnshuffle (must evenly divide height and width).
            d_model (int): Transformer embedding dimension.
            num_layers (int): Number of TransformerEncoder layers.
            nhead (int): Number of attention heads.
            height (int): Input image height.
            width (int): Input image width.
            dropout (float): Dropout rate used inside TransformerEncoder layers.
        """
        super(Model, self).__init__()

        assert height % downscale_factor == 0 and width % downscale_factor == 0, \
            "height and width must be divisible by downscale_factor"

        self.in_channels = in_channels
        self.downscale = downscale_factor
        self.d_model = d_model
        self.height = height
        self.width = width

        # After PixelUnshuffle, spatial dims are reduced
        self.h_small = height // downscale_factor
        self.w_small = width // downscale_factor

        # Number of channels after PixelUnshuffle: in_channels * (r^2)
        self.token_dim = in_channels * (downscale_factor ** 2)

        # PixelUnshuffle layer (part of available ops)
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor)

        # Unfold to extract per-location tokens (kernel_size=1 -> each spatial location becomes a token)
        self.unfold = nn.Unfold(kernel_size=1, stride=1)

        # Linear projections: token_dim -> d_model and back
        self.proj_in = nn.Linear(self.token_dim, d_model)
        self.proj_out = nn.Linear(d_model, self.token_dim)

        # Positional embedding for sequence length = h_small * w_small
        self.seq_len = self.h_small * self.w_small
        self.pos_embed = nn.Parameter(torch.randn(1, self.seq_len, d_model))

        # LayerNorm on embeddings (applied in (B, L, E) form)
        self.layernorm = nn.LayerNorm(d_model)

        # Transformer encoder stack (uses available nn.TransformerEncoder)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, dim_feedforward=4 * d_model, activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Small feed-forward convolution after reconstruction (1x1 conv) to allow mixing channels a bit
        self.post_conv = nn.Conv2d(self.token_dim, in_channels * (downscale_factor ** 2), kernel_size=1)

        # Output gate for residual blending
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Reconstructed tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        assert C == self.in_channels and H == self.height and W == self.width, \
            "Input tensor shape must match initialized dimensions"

        # 1) PixelUnshuffle: (B, C, H, W) -> (B, C*r^2, H/r, W/r)
        x_small = self.pixel_unshuffle(x)  # (B, token_dim, h_small, w_small)

        # 2) Unfold with kernel_size=1 to get (B, token_dim, L)
        x_unfold = self.unfold(x_small)  # (B, token_dim, L) where L = h_small * w_small

        # 3) Tokens as (B, L, token_dim)
        x_tokens = x_unfold.permute(0, 2, 1)  # (B, L, token_dim)

        # 4) Linear projection into transformer embedding space
        x_emb = self.proj_in(x_tokens)  # (B, L, d_model)

        # 5) Add positional embeddings and normalize
        x_emb = x_emb + self.pos_embed  # broadcast over batch
        x_emb = self.layernorm(x_emb)  # (B, L, d_model)

        # 6) Transformer expects (S, N, E) = (L, B, d_model)
        x_trans_in = x_emb.permute(1, 0, 2)  # (L, B, d_model)
        x_trans_out = self.transformer_encoder(x_trans_in)  # (L, B, d_model)

        # 7) Back to (B, L, d_model)
        x_trans = x_trans_out.permute(1, 0, 2)  # (B, L, d_model)

        # 8) Project back to token_dim
        x_proj = self.proj_out(x_trans)  # (B, L, token_dim)

        # 9) Prepare for folding: (B, token_dim, L)
        x_fold_input = x_proj.permute(0, 2, 1)  # (B, token_dim, L)

        # 10) Fold back to spatial layout of small image (h_small, w_small)
        x_small_recon = F.fold(x_fold_input, output_size=(self.h_small, self.w_small), kernel_size=1, stride=1)
        # x_small_recon shape: (B, token_dim, h_small, w_small)

        # 11) Optionally mix channels with a 1x1 conv and add residual connection
        x_small_mix = self.post_conv(x_small_recon)  # (B, token_dim, h_small, w_small)

        # Residual blending between original small representation and reconstructed one
        gated = torch.sigmoid(self.gate)
        x_small_combined = gated * x_small_mix + (1 - gated) * x_small  # (B, token_dim, h_small, w_small)

        # 12) Pixel shuffle to reconstruct original spatial resolution:
        # Use functional.pixel_shuffle to invert PixelUnshuffle (upscale_factor = downscale_factor)
        x_out = F.pixel_shuffle(x_small_combined, upscale_factor=self.downscale)  # (B, in_channels, H, W)

        return x_out


# Configuration variables
batch_size = 4
in_channels = 3
height = 128
width = 128
downscale_factor = 2  # must divide height and width
d_model = 256
nhead = 8
num_layers = 4
dropout = 0.1

def get_inputs():
    """
    Returns runtime input tensors for the model.
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization inputs for the Model constructor, in the same order as __init__.
    """
    return [in_channels, downscale_factor, d_model, num_layers, nhead, height, width, dropout]