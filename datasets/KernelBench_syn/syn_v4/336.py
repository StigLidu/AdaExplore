import torch
import torch.nn as nn
from typing import Tuple

class Model(nn.Module):
    """
    Complex 3D pooling + attention module:
    - Applies AdaptiveAvgPool3d to reduce spatial dims.
    - Flattens spatial grid into a sequence and linearly projects to an attention embedding.
    - Runs MultiheadAttention over the spatial tokens (self-attention).
    - Projects back to channel dimension, applies Softshrink non-linearity.
    - Aggregates tokens by mean to produce a per-batch, per-channel descriptor.
    """

    def __init__(
        self,
        in_channels: int,
        pool_output_size: Tuple[int, int, int],
        embed_dim: int,
        num_heads: int,
        softshrink_lambda: float = 0.5
    ):
        """
        Args:
            in_channels (int): Number of input channels (C).
            pool_output_size (Tuple[int,int,int]): Output (D,H,W) for AdaptiveAvgPool3d.
            embed_dim (int): Embedding dimension used by MultiheadAttention.
            num_heads (int): Number of attention heads (must divide embed_dim).
            softshrink_lambda (float, optional): Lambda parameter for Softshrink. Defaults to 0.5.
        """
        super(Model, self).__init__()

        # Pooling to fixed spatial grid
        self.avgpool3d = nn.AdaptiveAvgPool3d(pool_output_size)

        # Project channel vectors to attention embedding dimension
        self.to_embed = nn.Linear(in_channels, embed_dim)

        # Multi-head self-attention operating on spatial tokens (batch_first=True)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

        # Project attention outputs back to original channel dimension
        self.to_channels = nn.Linear(embed_dim, in_channels)

        # Element-wise soft shrink activation
        self.softshrink = nn.Softshrink(lambd=softshrink_lambda)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, depth, height, width).

        Returns:
            torch.Tensor: Aggregated descriptor of shape (batch_size, channels).
        """
        # 1) Adaptive average pool to reduce spatial dims to a fixed grid
        pooled = self.avgpool3d(x)  # (B, C, d', h', w')

        # 2) Flatten spatial grid into sequence of tokens and put channels as features
        B, C, d, h, w = pooled.shape
        S = d * h * w
        tokens = pooled.view(B, C, S).permute(0, 2, 1)  # (B, S, C)

        # 3) Linear projection to attention embedding
        embed = self.to_embed(tokens)  # (B, S, E)

        # 4) Self-attention over spatial tokens
        attn_out, _ = self.attn(embed, embed, embed)  # (B, S, E)

        # 5) Project back to channel space
        channels_out = self.to_channels(attn_out)  # (B, S, C)

        # 6) Non-linear shrinkage
        shrunk = self.softshrink(channels_out)  # (B, S, C)

        # 7) Aggregate tokens by mean to get a per-channel descriptor
        aggregated = shrunk.mean(dim=1)  # (B, C)

        return aggregated

# Configuration / default sizes
batch_size = 8
channels = 64
depth = 16
height = 32
width = 32

# Pool output size (D', H', W') after AdaptiveAvgPool3d
pool_d = 4
pool_h = 4
pool_w = 4

# Attention hyperparameters (embed_dim must be divisible by num_heads)
embed_dim = 32
num_heads = 4
softshrink_lambda = 0.3

def get_inputs():
    """
    Returns:
        List[torch.Tensor]: List containing a single input tensor of shape
                            (batch_size, channels, depth, height, width).
    """
    x = torch.randn(batch_size, channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Returns:
        List: Initialization arguments for the Model constructor in the same order:
              in_channels, pool_output_size, embed_dim, num_heads, softshrink_lambda
    """
    return [channels, (pool_d, pool_h, pool_w), embed_dim, num_heads, softshrink_lambda]