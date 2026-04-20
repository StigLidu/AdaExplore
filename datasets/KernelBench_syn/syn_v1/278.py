import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Image-to-vector model that:
    - Projects spatial features into a token sequence via a 1x1 convolution.
    - Adds a small learnable positional embedding broadcasted across tokens.
    - Applies MultiheadAttention over the spatial token sequence (self-attention).
    - Uses a residual connection followed by Softplus activation.
    - Reprojects tokens back to a spatial map and applies MaxPool2d.
    - Produces a final vector per-batch by global-average-pooling the pooled feature map.

    This combines convolutional projection, transformer-style attention, non-linear activation,
    and pooling to create a heterogeneous computation pattern.
    """
    def __init__(self, in_channels: int, embed_dim: int, num_heads: int, pool_kernel: int = 2):
        """
        Args:
            in_channels (int): Number of input channels of the image.
            embed_dim (int): Embedding dimensionality for attention (must be divisible by num_heads).
            num_heads (int): Number of attention heads.
            pool_kernel (int): Kernel size for MaxPool2d.
        """
        super(Model, self).__init__()
        # 1x1 convolution to map image channels to attention embedding dimension
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        # Multi-head self-attention; use batch_first to work with (B, seq_len, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        # Non-linear activation after residual addition
        self.softplus = nn.Softplus()
        # Spatial pooling to reduce H/W after reprojecting tokens back to spatial map
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_kernel)
        # A small learnable positional embedding vector (broadcast across spatial tokens)
        # We allocate a single position vector and expand it to sequence length at runtime.
        self.pos_emb = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, in_channels, height, width)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, embed_dim) representing pooled features.
        """
        # x -> projected spatial features
        # proj: (B, embed_dim, H, W)
        proj = self.proj(x)

        B, E, H, W = proj.shape
        # Flatten spatial dims to sequence: (B, seq_len, embed_dim)
        seq = proj.flatten(2).transpose(1, 2)  # (B, H*W, E)

        # Broadcast the small positional embedding across the token sequence
        pos = self.pos_emb.expand(B, seq.size(1), -1)  # (B, seq_len, E)
        tokens = seq + pos

        # Self-attention over spatial tokens
        attn_out, _attn_weights = self.attn(tokens, tokens, tokens)  # (B, seq_len, E)

        # Residual connection + non-linearity
        tokens = self.softplus(attn_out + tokens)  # (B, seq_len, E)

        # Reproject tokens back to spatial map: (B, E, H, W)
        spatial = tokens.transpose(1, 2).view(B, E, H, W)

        # Spatial max pooling reduces H and W
        pooled = self.pool(spatial)  # (B, E, H//k, W//k)

        # Global average pooling over spatial dimensions to produce a single vector per batch
        out = pooled.mean(dim=(2, 3))  # (B, E)
        return out

# Configuration variables
batch_size = 8
in_channels = 3
height = 64
width = 64
embed_dim = 64   # must be divisible by num_heads
num_heads = 8
pool_kernel = 2

def get_inputs():
    """
    Returns a list containing a single input image tensor.
    Shape: (batch_size, in_channels, height, width)
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters for the Model constructor:
    [in_channels, embed_dim, num_heads, pool_kernel]
    """
    return [in_channels, embed_dim, num_heads, pool_kernel]