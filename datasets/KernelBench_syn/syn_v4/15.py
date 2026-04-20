import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

class Model(nn.Module):
    """
    Complex model combining AdaptiveMaxPool2d, a TransformerEncoder stack, and LeakyReLU.
    Pipeline:
      - Adaptive max-pool the spatial dimensions to a fixed (h_out, w_out).
      - Flatten spatial grid to a sequence of tokens (seq_len = h_out * w_out).
      - Linear projection from input channels -> d_model.
      - Add learned positional embeddings.
      - Pass through a stack of TransformerEncoder layers.
      - Project transformer token outputs back to input channel dimension.
      - Reshape tokens back to spatial grid and upsample to original image size.
      - Apply LeakyReLU activation on reconstructed image.
    """
    def __init__(
        self,
        in_channels: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        adaptive_output_size: Tuple[int, int],
        upsample_mode: str = "bilinear",
        negative_slope: float = 0.1,
    ):
        """
        Args:
            in_channels: Number of channels in the input images.
            d_model: Transformer embedding dimension.
            nhead: Number of attention heads (must divide d_model).
            num_layers: Number of TransformerEncoder layers.
            dim_feedforward: Hidden dimension in Transformer feedforward layers.
            adaptive_output_size: Target (h_out, w_out) after AdaptiveMaxPool2d.
            upsample_mode: Mode used by F.interpolate to upsample back to original size.
            negative_slope: Negative slope parameter for LeakyReLU.
        """
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.h_out, self.w_out = adaptive_output_size
        self.seq_len = self.h_out * self.w_out
        self.upsample_mode = upsample_mode

        # Adaptive pooling to reduce spatial dimensions to fixed grid
        self.pool = nn.AdaptiveMaxPool2d((self.h_out, self.w_out))

        # Project per-location channel vector into Transformer embedding space
        self.input_proj = nn.Linear(in_channels, d_model)

        # Learned positional embeddings for each grid location (seq_len, d_model)
        self.pos_embed = nn.Parameter(torch.randn(self.seq_len, d_model))

        # Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Project transformer outputs back to original channel dimension
        self.output_proj = nn.Linear(d_model, in_channels)

        # Non-linear activation applied to reconstructed image
        self.leaky = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input image tensor of shape (batch_size, in_channels, H, W)

        Returns:
            Reconstructed image tensor of shape (batch_size, in_channels, H, W)
            produced by processing through pooling -> transformer -> projection -> upsampling.
        """
        # Save original spatial size for upsampling later
        B, C, H, W = x.shape

        # 1) Adaptive max pooling to fixed grid (B, C, h_out, w_out)
        pooled = self.pool(x)

        # 2) Flatten spatial grid to sequence tokens: (B, seq_len, C)
        # reorder to (B, h_out, w_out, C) then collapse spatial dims
        tokens = pooled.permute(0, 2, 3, 1).contiguous().view(B, self.seq_len, C)

        # 3) Linear projection to d_model: (B, seq_len, d_model)
        tokens = self.input_proj(tokens)

        # 4) Add positional embeddings (broadcast over batch)
        tokens = tokens + self.pos_embed.unsqueeze(0)  # (1, seq_len, d_model) -> broadcast

        # 5) Transformer expects (S, N, E) = (seq_len, batch, d_model)
        tokens = tokens.permute(1, 0, 2).contiguous()  # (seq_len, B, d_model)

        # 6) Pass through TransformerEncoder
        encoded = self.transformer(tokens)  # (seq_len, B, d_model)

        # 7) Back to (B, seq_len, d_model)
        encoded = encoded.permute(1, 0, 2).contiguous()

        # 8) Project back to channel space per-token: (B, seq_len, C)
        recon_tokens = self.output_proj(encoded)

        # 9) Reshape tokens back to spatial grid (B, C, h_out, w_out)
        recon = recon_tokens.view(B, self.h_out, self.w_out, C).permute(0, 3, 1, 2).contiguous()

        # 10) Upsample back to original (H, W)
        recon = F.interpolate(recon, size=(H, W), mode=self.upsample_mode, align_corners=False)

        # 11) Final non-linearity
        recon = self.leaky(recon)

        return recon

# Module-level configuration variables
batch_size = 8
in_channels = 64
height = 64
width = 48

# Transformer / pooling configuration
adaptive_output_size = (8, 6)   # h_out, w_out -> seq_len = 48
d_model = 128
nhead = 8                       # must divide d_model
num_layers = 4
dim_feedforward = 512

def get_inputs() -> List[torch.Tensor]:
    """
    Create and return typical input tensors for the model.
    Returns:
        [x]: single-element list with an input image tensor of shape (batch_size, in_channels, height, width)
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs() -> List:
    """
    Return the initialization parameters for the Model constructor, in the same order as __init__ signature.
    """
    # Note: Model.__init__ signature:
    # def __init__(self, in_channels, d_model, nhead, num_layers, dim_feedforward, adaptive_output_size, ...)
    return [in_channels, d_model, nhead, num_layers, dim_feedforward, adaptive_output_size]