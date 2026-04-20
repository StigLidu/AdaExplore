import math
import torch
import torch.nn as nn

# Configuration
BATCH_SIZE = 8
IN_CHANNELS = 3        # actual in_channels used in get_inputs; LazyConv1d will infer this
SEQ_LEN = 1024         # sequence length; chosen to be a perfect square (32 x 32)
OUT_CHANNELS = 64      # number of output channels for the conv layer
POOL_OUT = (8, 8)      # adaptive pool output spatial dims
EMBED_DIM = 512        # final embedding dimension

class Model(nn.Module):
    """
    A composite model that:
    - Applies a LazyConv1d to a 3-channel sequence input (lazy in_channels inferred at first forward)
    - Reshapes the conv output into a 2D feature map (treating sequence length as H*W)
    - Applies AdaptiveAvgPool2d to reduce spatial resolution
    - Flattens and projects pooled features to an embedding dimension
    - Adds a residual projection obtained by global-average-pooling the conv sequence features
    - Uses GELU activation before returning the final embedding

    The model demonstrates mixing 1D convolution, 2D pooling, nonlinear activation, and
    both spatial and channel-wise aggregation patterns.
    """
    def __init__(self,
                 out_channels: int = OUT_CHANNELS,
                 pool_out: tuple = POOL_OUT,
                 embed_dim: int = EMBED_DIM):
        super(Model, self).__init__()
        # LazyConv1d will infer in_channels from the first forward input
        self.conv = nn.LazyConv1d(out_channels=out_channels, kernel_size=5, padding=2)
        # AdaptiveAvgPool2d to reduce the 2D map to a fixed spatial size
        self.pool2d = nn.AdaptiveAvgPool2d(pool_out)
        # Non-linear activation
        self.gelu = nn.GELU()
        # Projection from pooled flattened features to embedding
        pooled_flat_size = out_channels * pool_out[0] * pool_out[1]
        self.proj = nn.Linear(pooled_flat_size, embed_dim)
        # Residual projection from channel-global features to embedding
        self.residual_proj = nn.Linear(out_channels, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (N, C_in, L) where L should be a perfect square.
                              C_in will be inferred for the LazyConv1d on first call.

        Returns:
            torch.Tensor: Output tensor of shape (N, embed_dim)
        """
        # x: (N, C_in, L)
        N = x.size(0)

        # 1) 1D convolution along the sequence dimension -> (N, out_channels, L)
        conv_out = self.conv(x)

        # 2) Prepare residual: global average pool over the length dimension -> (N, out_channels)
        residual = conv_out.mean(dim=2)  # channel-wise summary

        # 3) Interpret sequence length L as a square 2D map: L = side * side
        L = conv_out.size(2)
        side = int(math.isqrt(L))
        if side * side != L:
            raise ValueError(f"Sequence length (conv output length) must be a perfect square, got L={L}")
        # Reshape to (N, out_channels, side, side)
        feature_map = conv_out.view(N, conv_out.size(1), side, side)

        # 4) Adaptive average pooling to fixed spatial resolution -> (N, out_channels, pool_h, pool_w)
        pooled = self.pool2d(feature_map)

        # 5) Flatten pooled spatial dims and project -> (N, embed_dim)
        pooled_flat = pooled.view(N, -1)
        projected = self.proj(pooled_flat)

        # 6) Residual projection and combine with nonlinear activation
        residual_emb = self.residual_proj(residual)
        out = self.gelu(projected) + residual_emb

        return out

def get_inputs():
    """
    Returns a list containing a single input tensor matching the expected shape:
    (BATCH_SIZE, IN_CHANNELS, SEQ_LEN)
    """
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, SEQ_LEN)
    return [x]

def get_init_inputs():
    """
    No special initialization inputs required (LazyConv1d infers in_channels from data).
    """
    return []