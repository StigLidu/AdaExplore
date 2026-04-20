import torch
import torch.nn as nn

"""
Complex 3D processing module combining Conv3d, GroupNorm, Tanhshrink activation,
Dropout3d, AlphaDropout and two projection heads with a global pooling + residual fusion.

Structure:
- Model(nn.Module)
  - __init__ takes architecture hyperparameters
  - forward runs the computation pipeline described below

Computation pipeline (forward):
1. conv3d -> groupnorm
2. element-wise Tanhshrink nonlinearity
3. channel-wise dropout (Dropout3d)
4. global average pooling to (B, C_mid)
5. alpha dropout on pooled features
6. two projection heads:
   - main head: maps pooled mid-channels -> out_features
   - skip head: pools original input channels and maps -> out_features
7. fuse heads (sum), apply a final scaling (sigmoid) and return
"""

# Configuration variables (module-level)
batch_size = 8
in_channels = 16
mid_channels = 32
out_features = 128
depth = 8
height = 8
width = 6
dropout3d_p = 0.2
alpha_dropout_p = 0.05
group_norm_groups = 4

class Model(nn.Module):
    """
    Complex 3D model that demonstrates a variety of operations:
    - 3D convolution + GroupNorm
    - Tanhshrink activation (element-wise)
    - Dropout3d (channel-wise dropout)
    - Global pooling -> AlphaDropout -> fully-connected projection
    - Skip connection from pooled input channels projected and fused
    """
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_features: int,
        dropout3d_p: float = 0.2,
        alpha_dropout_p: float = 0.05,
        gn_groups: int = 4,
    ):
        super(Model, self).__init__()

        # First projection from input channels into a richer feature space
        self.conv3d = nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.gn = nn.GroupNorm(num_groups=gn_groups, num_channels=mid_channels)
        self.tanhshrink = nn.Tanhshrink()          # elementwise nonlinearity
        self.dropout3d = nn.Dropout3d(p=dropout3d_p)  # zeros whole channels randomly

        # Global pooling to produce compact descriptors
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # AlphaDropout applied to pooled features (suitable for preserving self-normalizing activations)
        self.alpha_dropout = nn.AlphaDropout(p=alpha_dropout_p)

        # Two projection heads that produce the final output vector:
        # - main head from mid_channels pooled features
        # - skip head from original input-channels pooled features (residual info)
        self.main_fc = nn.Linear(mid_channels, out_features, bias=True)
        self.skip_fc = nn.Linear(in_channels, out_features, bias=True)

        # Final fusion scaling (small learnable scalar)
        self.fusion_scale = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C_in, D, H, W)

        Returns:
            out: Tensor of shape (B, out_features)
        """
        # Save original for skip pooling
        orig = x

        # Conv -> Norm -> Tanhshrink -> Dropout3d
        y = self.conv3d(x)             # (B, mid_channels, D, H, W)
        y = self.gn(y)
        y = self.tanhshrink(y)
        y = self.dropout3d(y)

        # Global pooling to (B, mid_channels)
        y_pooled = self.global_pool(y)                       # (B, mid, 1, 1, 1)
        y_pooled = y_pooled.view(y_pooled.size(0), -1)      # (B, mid_channels)

        # AlphaDropout on pooled features
        y_pooled = self.alpha_dropout(y_pooled)

        # Main projection
        main_out = self.main_fc(y_pooled)                   # (B, out_features)

        # Skip connection: pool original input channels and project
        skip_pooled = self.global_pool(orig).view(orig.size(0), -1)  # (B, in_channels)
        skip_out = self.skip_fc(skip_pooled)                         # (B, out_features)

        # Fuse projections with learnable scale and non-linear gating
        fused = main_out + self.fusion_scale * skip_out
        # Final non-linearity to bound outputs (use sigmoid to map into (0,1))
        out = torch.sigmoid(fused)

        return out

def get_inputs():
    """
    Returns inputs for a forward pass:
    - A random 5D tensor with shape (batch_size, in_channels, depth, height, width)
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization inputs for the Model constructor:
    [in_channels, mid_channels, out_features, dropout3d_p, alpha_dropout_p, group_norm_groups]
    """
    return [in_channels, mid_channels, out_features, dropout3d_p, alpha_dropout_p, group_norm_groups]