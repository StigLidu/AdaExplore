import torch
import torch.nn as nn

# Module-level configuration
BATCH = 8
CHANNELS = 64
DEPTH = 16
HEIGHT = 64
WIDTH = 64

# Adaptive pool target (D_out, H_out, W_out)
POOL_OUTPUT = (4, 8, 8)

# Dropout probability for Dropout2d
DROPOUT_P = 0.25

# Output feature dimension for final projection
OUT_FEATURES = 512


class Model(nn.Module):
    """
    A moderately complex module that demonstrates a 3D-to-2D pooling + channel fusion pattern,
    followed by spatial-pointwise projection using LazyLinear and a small global residual path.

    Computation summary:
    - AdaptiveAvgPool3d reduces spatial/temporal resolution to a fixed (D', H', W')
    - Depth dimension is folded into channels to produce a (N, C * D', H', W') tensor
    - nn.Dropout2d randomly drops entire channels (works on the combined channels)
    - The tensor is rearranged so each spatial location is a separate vector and passed through
      an nn.LazyLinear layer (in_features inferred on first forward) to produce per-location embeddings
    - Per-location embeddings are averaged to produce a global feature vector (N, out_features)
    - A small residual branch computes a global channel-wise average of the original input and
      projects it to out_features using a standard nn.Linear
    - The two paths are combined and normalized with LayerNorm
    """
    def __init__(self,
                 channels: int,
                 pool_output: tuple,
                 dropout_p: float,
                 out_features: int):
        """
        Args:
            channels (int): Number of input channels C for the (B, C, D, H, W) input.
            pool_output (tuple): Target output size for AdaptiveAvgPool3d (D_out, H_out, W_out).
            dropout_p (float): Probability for Dropout2d.
            out_features (int): Output feature dimension for the LazyLinear projection.
        """
        super(Model, self).__init__()
        # 3D adaptive average pooling to compress D/H/W to fixed sizes
        self.pool3d = nn.AdaptiveAvgPool3d(pool_output)

        # Drop entire channels after folding depth into channel dimension
        self.dropout2d = nn.Dropout2d(p=dropout_p)

        # LazyLinear will infer in_features when we first pass a tensor of shape (..., in_features).
        # We use it to project per-spatial-location vectors (channel-fused) to out_features.
        self.fc = nn.LazyLinear(out_features)

        # Residual projection: project original per-channel global averages to out_features.
        # channels is known at init, so a regular Linear is appropriate here.
        self.res_proj = nn.Linear(channels, out_features)

        # Normalize final combined representation
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, D, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (B, out_features)
        """
        # x: (B, C, D, H, W)
        B = x.size(0)

        # 1) Adaptive average pool to reduce the 3D spatial dims to fixed size
        # pooled: (B, C, Dp, Hp, Wp)
        pooled = self.pool3d(x)

        # 2) Fold depth into channels -> (B, C * Dp, Hp, Wp)
        B, C, Dp, Hp, Wp = pooled.shape
        merged = pooled.view(B, C * Dp, Hp, Wp)

        # 3) Channel-wise dropout (Dropout2d operates on channels of 4D tensors)
        dropped = self.dropout2d(merged)

        # 4) Rearrange so that each spatial location across Hp x Wp becomes a vector to be projected
        #    from (B, C*Dp, Hp, Wp) -> (B, Hp, Wp, C*Dp)
        per_loc = dropped.permute(0, 2, 3, 1).contiguous()

        # 5) Flatten batch & spatial dims so LazyLinear can operate on last dim and infer in_features
        #    flat: (B * Hp * Wp, C * Dp)
        flat = per_loc.view(-1, per_loc.shape[-1])

        # 6) Project each spatial location to out_features -> (B * Hp * Wp, out_features)
        projected = self.fc(flat)

        # 7) Reshape back to spatial layout -> (B, Hp, Wp, out_features)
        spatial_feats = projected.view(B, Hp, Wp, -1)

        # 8) Global average over spatial locations to get a compact global descriptor -> (B, out_features)
        global_feat = spatial_feats.mean(dim=(1, 2))

        # 9) Residual branch: global average of original input across D/H/W per channel -> (B, C)
        global_orig = x.mean(dim=(2, 3, 4))

        # 10) Project residual to out_features -> (B, out_features)
        residual = self.res_proj(global_orig)

        # 11) Combine and normalize
        out = self.norm(global_feat + residual)

        return out


def get_inputs():
    """
    Returns:
        A list containing a single input tensor of shape (BATCH, CHANNELS, DEPTH, HEIGHT, WIDTH)
    """
    x = torch.randn(BATCH, CHANNELS, DEPTH, HEIGHT, WIDTH)
    return [x]


def get_init_inputs():
    """
    Returns initialization parameters for the Model constructor in the same order:
    (channels, pool_output, dropout_p, out_features)
    """
    return [CHANNELS, POOL_OUTPUT, DROPOUT_P, OUT_FEATURES]