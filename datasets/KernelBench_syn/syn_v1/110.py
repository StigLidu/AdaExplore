import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex 3D feature-transform module that demonstrates a multi-stage computation:
      1. FeatureAlphaDropout to stochastically mask entire features (channels).
      2. Adaptive average pooling to reduce spatial resolution.
      3. Two fully-connected projections (bottleneck) with ReLU non-linearity.
      4. Reshape and trilinear upsampling back to the original spatial resolution.
      5. Dropout3d to randomly zero whole channels after upsampling.
      6. Residual connection and final ReLU activation.

    This pattern combines dropout variants and activations in a channel-aware 3D pipeline
    and is functionally distinct from simple pooling, bmm, or single-activation examples.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        pool_output: tuple = (4, 4, 4),
        dropout_p: float = 0.2,
        feature_dropout_p: float = 0.1
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            hidden_dim (int): Hidden dimensionality for the bottleneck FC layer.
            pool_output (tuple): (d, h, w) target size for adaptive pooling.
            dropout_p (float): Probability for nn.Dropout3d.
            feature_dropout_p (float): Probability for nn.FeatureAlphaDropout.
        """
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.pool_output = pool_output  # (pd, ph, pw)

        # Dropout that preserves self-normalizing properties (feature-wise)
        self.feature_dropout = nn.FeatureAlphaDropout(feature_dropout_p)
        # Channel dropout after upsampling
        self.dropout3d = nn.Dropout3d(p=dropout_p)
        self.relu = nn.ReLU()

        # Adaptive pooling to reduce spatial resolution to pool_output
        self.pool = nn.AdaptiveAvgPool3d(self.pool_output)

        # Fully connected bottleneck operating on flattened (C * pd * ph * pw)
        flattened = in_channels * self.pool_output[0] * self.pool_output[1] * self.pool_output[2]
        self.fc1 = nn.Linear(flattened, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, flattened)

        # Note: no parameters for interpolation; use functional.interpolate in forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of same shape as input.
        """
        # Preserve identity for residual connection
        identity = x

        # 1) FeatureAlphaDropout: stochastic feature/channel masking preserving normalization
        x = self.feature_dropout(x)

        # 2) Spatial dimensionality reduction via adaptive avg pooling
        pooled = self.pool(x)  # shape: (B, C, pd, ph, pw)

        # 3) Flatten and project through FC bottleneck with ReLU
        B = pooled.size(0)
        flat = pooled.view(B, -1)  # shape: (B, C * pd * ph * pw)
        bottleneck = self.fc1(flat)
        activated = self.relu(bottleneck)
        expanded = self.fc2(activated)

        # 4) Reshape back to (B, C, pd, ph, pw)
        reshaped = expanded.view(B, self.in_channels, self.pool_output[0], self.pool_output[1], self.pool_output[2])

        # 5) Upsample (trilinear) back to original spatial dims
        orig_d, orig_h, orig_w = identity.shape[2], identity.shape[3], identity.shape[4]
        upsampled = F.interpolate(reshaped, size=(orig_d, orig_h, orig_w), mode='trilinear', align_corners=False)

        # 6) Channel-wise dropout and residual addition followed by final ReLU
        dropped = self.dropout3d(upsampled)
        out = self.relu(dropped + identity)

        return out

# Configuration variables (module level)
batch_size = 8
channels = 32
depth = 32
height = 48
width = 48

hidden_dim = 4096
pool_output = (4, 6, 6)  # reduced spatial footprint for the bottleneck
dropout_p = 0.25
feature_dropout_p = 0.12

def get_inputs():
    """
    Returns example input tensors matching the Model forward signature.
    """
    x = torch.randn(batch_size, channels, depth, height, width, dtype=torch.float32)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters that should be used to construct the Model:
    (in_channels, hidden_dim, pool_output, dropout_p, feature_dropout_p)
    """
    return [channels, hidden_dim, pool_output, dropout_p, feature_dropout_p]