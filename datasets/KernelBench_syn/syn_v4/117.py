import torch
import torch.nn as nn

"""
Complex 3D spatial-channel mixing module demonstrating:
- AlphaDropout
- LazyBatchNorm3d (lazy-initialized on first forward)
- Threshold activation
- Learned channel projection and learned spatial mixing via einsum operations

Structure matches provided examples:
- Model class inheriting from nn.Module
- __init__ with configurable parameters
- forward doing a sequence of operations
- get_inputs() producing sample inputs
- get_init_inputs() returning initialization parameters
- Module-level configuration variables
"""

# Configuration (input shapes)
BATCH = 2
C = 16        # Number of channels
D = 4         # Depth
H = 4         # Height
W = 4         # Width

# Derived spatial size
SPATIAL_SIZE = D * H * W


class Model(nn.Module):
    """
    3D model that:
    - Applies AlphaDropout to the input
    - Projects channel dimension with a learned matrix
    - Applies LazyBatchNorm3d (initialized on first forward)
    - Threshold activation to clamp small values
    - Performs learned spatial mixing across flattened D*H*W positions
    - Adds a small residual connection from the (scaled) original input
    """

    def __init__(self, dropout_prob: float = 0.15, threshold: float = 0.05):
        """
        Args:
            dropout_prob: dropout probability for AlphaDropout
            threshold: threshold value for nn.Threshold; values <= threshold become 0.0
        """
        super(Model, self).__init__()

        # Dropout layer (AlphaDropout)
        self.dropout = nn.AlphaDropout(p=dropout_prob)

        # Lazy batch norm for 3D inputs; num_features will be inferred on first forward
        self.bn3d = nn.LazyBatchNorm3d()

        # Threshold activation: values <= threshold replaced by 0.0
        self.thresh = nn.Threshold(threshold, 0.0)

        # Learned channel projection: C_in x C_out (here square projection)
        # Use small initialization scale
        self.channel_proj = nn.Parameter(torch.randn(C, C) * 0.02)

        # Channel bias
        self.channel_bias = nn.Parameter(torch.zeros(C))

        # Learned spatial mixing matrix across flattened spatial dimension (S x S)
        self.spatial_mix = nn.Parameter(torch.randn(SPATIAL_SIZE, SPATIAL_SIZE) * 0.02)

        # Small scaling for residual path
        self.residual_scale = 0.125

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (N, C, D, H, W)

        Returns:
            Tensor of same shape (N, C, D, H, W) after the sequence of operations.
        """
        # Preserve a (small) residual from the original input
        residual = x * self.residual_scale  # (N, C, D, H, W)

        # 1) AlphaDropout (stochastic regularization)
        out = self.dropout(x)

        # 2) Channel projection (learned linear mixing across channels)
        #    Use einsum to multiply channels by the projection matrix:
        #    out: (N, C_in, D, H, W), channel_proj: (C_in, C_out) -> output (N, C_out, D, H, W)
        out = torch.einsum('n c d h w, c o -> n o d h w', out, self.channel_proj)

        # Add channel bias (broadcast over spatial dims)
        out = out + self.channel_bias.view(1, -1, 1, 1, 1)

        # 3) Lazy BatchNorm3d (will initialize num_features to C on first call)
        out = self.bn3d(out)

        # 4) Threshold activation to zero-out small values
        out = self.thresh(out)

        # 5) Learned spatial mixing: flatten spatial dims and mix positions via matrix multiplication
        #    out_flat: (N, C, S)
        N = out.shape[0]
        out_flat = out.view(N, C, SPATIAL_SIZE)  # (N, C, S)
        # spatial_mix: (S, S) -> perform einsum to mix positions
        out_mixed = torch.einsum('n c s, s t -> n c t', out_flat, self.spatial_mix)  # (N, C, S)

        # reshape back to (N, C, D, H, W)
        out = out_mixed.view(N, C, D, H, W)

        # 6) Add residual
        out = out + residual

        return out


def get_inputs():
    """
    Returns a list containing a single input tensor of shape (BATCH, C, D, H, W)
    with normally distributed values.
    """
    x = torch.randn(BATCH, C, D, H, W)
    return [x]


def get_init_inputs():
    """
    Returns the default initialization inputs for the Model constructor:
    [dropout_prob, threshold]
    """
    return [0.15, 0.05]