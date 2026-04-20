import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Any

"""
Complex 3D processing module that combines ReflectionPad3d, Conv3d, BatchNorm3d,
AdaptiveMaxPool3d, FeatureAlphaDropout and a small projection path. The model
demonstrates a multi-path computation: main conv -> activation -> adaptive pooling
-> dropout -> linear projection, plus a skip projection via 1x1 conv and global
average pooling. The two paths are fused and a final activation is applied.
"""

# Configuration variables (can be modified to generate different sizes)
batch_size = 8
in_channels = 4
mid_channels = 32
out_features = 128
D, H, W = 16, 16, 16  # spatial dimensions for 3D input
pad_tuple: Tuple[int, int, int, int, int, int] = (1, 1, 1, 1, 1, 1)  # ReflectionPad3d expects 6 ints
pool_output_size: Tuple[int, int, int] = (4, 4, 4)
dropout_p = 0.1

class Model(nn.Module):
    """
    3D feature processing model combining reflection padding, convolutional
    feature extraction, adaptive max pooling, feature-alpha dropout, and a
    projection skip-connection. Produces a compact vector per batch element.
    """
    def __init__(
        self,
        in_channels: int = in_channels,
        mid_channels: int = mid_channels,
        out_features: int = out_features,
        pool_size: Tuple[int, int, int] = pool_output_size,
        dropout_p: float = dropout_p,
        pad: Tuple[int, int, int, int, int, int] = pad_tuple,
    ):
        """
        Initializes the module components.

        Args:
            in_channels (int): Number of input channels.
            mid_channels (int): Number of intermediate convolution channels.
            out_features (int): Size of the final output feature vector.
            pool_size (tuple): Output size for AdaptiveMaxPool3d (d, h, w).
            dropout_p (float): Probability for FeatureAlphaDropout.
            pad (tuple): 6-tuple for ReflectionPad3d (D_left, D_right, H_left, H_right, W_left, W_right).
        """
        super(Model, self).__init__()

        # Reflection padding to allow conv3d with kernel_size=3 to keep spatial dims
        self.pad = nn.ReflectionPad3d(pad)

        # Main conv block: conv -> batchnorm -> activation
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm3d(mid_channels)

        # Adaptive pooling to reduce spatial dimensions to a fixed grid
        self.pool = nn.AdaptiveMaxPool3d(pool_size)

        # Channel-wise dropout variant suited for SELU / self-normalizing networks
        self.dropout = nn.FeatureAlphaDropout(p=dropout_p)

        # Fully connected projection from pooled flattened features to output vector
        pooled_elems = mid_channels * pool_size[0] * pool_size[1] * pool_size[2]
        self.fc = nn.Linear(pooled_elems, out_features)

        # Skip projection: 1x1 conv to map intermediate channels to out_features channels,
        # followed by global average pooling over spatial dims to produce (N, out_features)
        self.skip_conv = nn.Conv3d(in_channels=mid_channels, out_channels=out_features, kernel_size=1, bias=True)

        # Final non-linearity
        self.final_act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing a fused representation.

        Steps:
            1. Reflection pad the input (preserve details at boundaries).
            2. Apply 3D convolution, batch-norm and SELU activation.
            3. Adaptive max pool to fixed small spatial grid.
            4. Flatten pooled features and apply FeatureAlphaDropout.
            5. Linear projection to out_features.
            6. In parallel, project conv features via 1x1 conv and global-average-pool
               to produce a skip vector.
            7. Fuse the two vectors (add), apply final activation and return.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, D, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (N, out_features)
        """
        # 1. Reflection padding
        x_p = self.pad(x)

        # 2. Convolutional feature extraction
        conv_feat = self.conv(x_p)           # shape: (N, mid_channels, D, H, W) (spatial dims preserved by pad)
        conv_feat = self.bn(conv_feat)
        conv_feat = F.selu(conv_feat)        # SELU activation

        # 3. Adaptive pooling to compress spatial dims
        pooled = self.pool(conv_feat)        # shape: (N, mid_channels, pD, pH, pW)

        # 4. Flatten and dropout
        N = pooled.shape[0]
        flattened = pooled.view(N, -1)       # shape: (N, mid_channels * pD * pH * pW)
        dropped = self.dropout(flattened)    # FeatureAlphaDropout applied to flattened feature vectors

        # 5. Fully connected projection
        proj = self.fc(dropped)              # shape: (N, out_features)

        # 6. Skip projection path: 1x1 conv -> global avg pool over spatial dims -> squeeze
        skip = self.skip_conv(conv_feat)     # shape: (N, out_features, D, H, W)
        # Global average pooling across D,H,W
        skip = skip.mean(dim=[2, 3, 4])      # shape: (N, out_features)

        # 7. Fuse paths and final activation
        fused = proj + skip
        out = self.final_act(fused)

        return out


def get_inputs() -> List[torch.Tensor]:
    """
    Generates a representative input tensor for the model.

    Returns:
        List[torch.Tensor]: A single-element list containing a random tensor
                            shaped (batch_size, in_channels, D, H, W).
    """
    x = torch.randn(batch_size, in_channels, D, H, W)
    return [x]


def get_init_inputs() -> List[Any]:
    """
    Returns the initialization parameters for the Model constructor so that
    the model can be instantiated consistently for tests/benchmarks.

    Returns:
        List[Any]: [in_channels, mid_channels, out_features, pool_output_size, dropout_p, pad_tuple]
    """
    return [in_channels, mid_channels, out_features, pool_output_size, dropout_p, pad_tuple]