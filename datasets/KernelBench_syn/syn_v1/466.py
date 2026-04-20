import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    A composite model that:
      - Projects input feature maps to an embedding dimension via a 1x1 convolution
      - Applies 2D fractional max pooling to reduce spatial dimensions
      - Treats the pooled spatial locations as a sequence and applies multi-head self-attention
      - Adds a residual connection, applies LeakyReLU, projects back to original channels
      - Upsamples to the original spatial resolution

    Forward pass summary:
      x (B, C, H, W)
        -> proj (B, E, H, W)
        -> fractional pooling -> (B, E, Hp, Wp)
        -> reshape to sequence (B, L, E)
        -> MultiheadAttention (batch_first=True) -> (B, L, E)
        -> residual + LeakyReLU
        -> reshape back (B, E, Hp, Wp)
        -> 1x1 conv to channels (B, C, Hp, Wp)
        -> interpolate to original (H, W)
    """
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        num_heads: int,
        fm_kernel=(2, 2),
        fm_output_ratio=(0.5, 0.5),
    ):
        """
        Initializes the composite module.

        Args:
            in_channels (int): Number of input channels.
            embed_dim (int): Embedding dimension used by attention (must be divisible by num_heads).
            num_heads (int): Number of attention heads.
            fm_kernel (tuple): Kernel size for FractionalMaxPool2d.
            fm_output_ratio (tuple): Output ratio for FractionalMaxPool2d (fractions for height and width).
        """
        super(Model, self).__init__()

        # 1x1 convolution to lift feature channels to embedding dimension
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False)

        # Fractional max-pooling to reduce spatial size in a fractional manner
        # We request indices in case one wants to inspect them (returned by forward)
        self.pool = nn.FractionalMaxPool2d(kernel_size=fm_kernel, output_ratio=fm_output_ratio, return_indices=True)

        # Multi-head self-attention operating on the pooled spatial sequence
        # Use batch_first=True to accept (B, L, E) shaped sequences
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

        # Non-linear activation after attention + residual
        self.activation = nn.LeakyReLU(negative_slope=0.01, inplace=False)

        # Project back to original number of channels
        self.reproj = nn.Conv2d(embed_dim, in_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W) (same spatial size as input).
        """
        # Project channels -> embedding dimension
        proj = self.proj(x)  # (B, E, H, W)

        # Fractional pooling reduces spatial dims; returns (output, indices)
        pooled, indices = self.pool(proj)  # pooled: (B, E, Hp, Wp)

        B, E, Hp, Wp = pooled.shape

        # Convert spatial grid to a sequence for attention: (B, L, E)
        seq = pooled.flatten(2).permute(0, 2, 1)  # (B, L, E), where L = Hp * Wp

        # Multi-head self-attention (self-attend)
        attn_out, attn_weights = self.attn(seq, seq, seq)  # attn_out: (B, L, E)

        # Residual connection and non-linearity
        res = seq + attn_out
        activated = self.activation(res)  # (B, L, E)

        # Reshape back to spatial grid
        feat = activated.permute(0, 2, 1).contiguous().view(B, E, Hp, Wp)  # (B, E, Hp, Wp)

        # Reproject to original channels
        out = self.reproj(feat)  # (B, C, Hp, Wp)

        # Upsample to match the original spatial resolution
        out_up = F.interpolate(out, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)

        return out_up

# Configuration / default sizes for the example
BATCH = 8
IN_CHANNELS = 64
HEIGHT = 32
WIDTH = 32
EMBED_DIM = 128
NUM_HEADS = 8
FM_KERNEL = (2, 2)
FM_OUTPUT_RATIO = (0.5, 0.5)

def get_inputs():
    """
    Returns example input tensors for running the model.
    """
    x = torch.randn(BATCH, IN_CHANNELS, HEIGHT, WIDTH)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters for the Model constructor.
    """
    return (IN_CHANNELS, EMBED_DIM, NUM_HEADS, FM_KERNEL, FM_OUTPUT_RATIO)