import torch
import torch.nn as nn

"""
Complex model combining LazyBatchNorm2d, Hardswish, and MultiheadAttention.

Pattern:
- Input image tensor (B, C, H, W)
- LazyBatchNorm2d followed by Hardswish
- Patch extraction via unfold -> sequence of patch tokens
- Linear projection to embedding dimension
- Multihead self-attention over spatial tokens (batch_first=True)
- Simple feed-forward residual block
- Project tokens back to patch pixels and reconstruct image
- Final Hardswish activation
"""

# Configuration
batch_size = 8
in_channels = 3
height = 32
width = 32
patch_size = 4
embed_dim = 128
num_heads = 8

class Model(nn.Module):
    """
    Image-to-image transformer-like block:
    - Normalizes and activates inputs (LazyBatchNorm2d + Hardswish)
    - Converts image patches into token sequence
    - Applies MultiheadAttention and an FFN with residual connections
    - Reconstructs the image from processed patches
    """
    def __init__(self,
                 in_channels: int = in_channels,
                 patch_size: int = patch_size,
                 embed_dim: int = embed_dim,
                 num_heads: int = num_heads):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # LazyBatchNorm2d will infer num_features on the first forward pass
        self.bn = nn.LazyBatchNorm2d()
        self.act = nn.Hardswish()

        # Project flattened patch (C * p * p) -> embed_dim
        self.patch_dim = in_channels * patch_size * patch_size
        self.proj = nn.Linear(self.patch_dim, embed_dim)

        # Multihead attention over spatial tokens (batch_first=True expects (B, S, E))
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

        # Lightweight feed-forward network with Hardswish non-linearity
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.Hardswish(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

        # Project back from embedding to patch pixels
        self.out_proj = nn.Linear(embed_dim, self.patch_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input image tensor of shape (B, C, H, W), where H and W are divisible by patch_size.

        Returns:
            Reconstructed image tensor of same shape (B, C, H, W).
        """
        # Normalize and activate
        x = self.bn(x)      # Lazy init will set num_features from x.shape[1] on first call
        x = self.act(x)

        B, C, H, W = x.shape
        p = self.patch_size
        assert H % p == 0 and W % p == 0, "Height and Width must be divisible by patch_size"

        # Extract non-overlapping patches: result shape after unfold (B, C, H/p, W/p, p, p)
        x_unf = x.unfold(2, p, p).unfold(3, p, p)  # (B, C, H/p, W/p, p, p)
        # Rearrange to (B, H/p * W/p, C * p * p)
        x_unf = x_unf.permute(0, 2, 3, 1, 4, 5).contiguous()
        num_h = x_unf.size(1)
        num_w = x_unf.size(2)
        seq_len = num_h * num_w
        patches = x_unf.view(B, seq_len, C * p * p)  # (B, S, patch_dim)

        # Project patches to embedding space
        tokens = self.proj(patches)  # (B, S, E)

        # Self-attention
        attn_out, _ = self.attn(tokens, tokens, tokens)  # (B, S, E)

        # Residual connection + FFN with residual
        tokens = tokens + attn_out
        tokens = tokens + self.ffn(tokens)

        # Project back to patch pixels and reconstruct the image
        patches_out = self.out_proj(tokens)  # (B, S, patch_dim)
        patches_out = patches_out.view(B, num_h, num_w, C, p, p)
        # Permute back to (B, C, H, W)
        out = patches_out.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, num_h * p, num_w * p)

        out = self.act(out)
        return out

def get_inputs():
    """
    Returns a list of input tensors for the model.
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters used to construct the Model instance.
    """
    return [in_channels, patch_size, embed_dim, num_heads]