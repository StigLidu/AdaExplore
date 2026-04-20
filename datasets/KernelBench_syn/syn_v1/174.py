import torch
import torch.nn as nn
from typing import List, Tuple

# Configuration
batch_size = 8
channels = 64
depth = 8
height = 32
width = 32

# Adaptive pooling output (depth must be 1 for the computation below since we squeeze that dimension)
adaptive_output = (1, 8, 8)  # (D_out, H_out, W_out)
avgpool2d_kernel = 2  # kernel and stride for AvgPool2d

class Model(nn.Module):
    """
    A moderately complex 3D -> 2D feature extractor and projector.
    Pipeline:
      1. BatchNorm3d over the input (N, C, D, H, W)
      2. AdaptiveAvgPool3d to reduce spatial dims -> (N, C, 1, H', W')
      3. Squeeze depth dimension -> (N, C, H', W')
      4. AvgPool2d to further downsample spatial dims -> (N, C, H2, W2)
      5. Flatten spatial dims and permute to (N, S, C) where S = H2 * W2
      6. Linear mixing via a learned parameter (matmul) -> (N, S, C)
      7. Spatial aggregation (mean over S) -> (N, C)
      8. Second projection via same mixing parameter -> (N, C)
      9. Residual add + nonlinearity (sigmoid) -> final (N, C)
    """
    def __init__(self,
                 channels: int,
                 adaptive_out: Tuple[int, int, int] = adaptive_output,
                 avg2d_kernel: int = avgpool2d_kernel):
        super(Model, self).__init__()
        D_out, H_out, W_out = adaptive_out
        assert D_out == 1, "This model squeezes the depth dimension after adaptive pooling; set D_out=1."

        self.channels = channels
        self.adaptive_out = adaptive_out
        self.avg2d_kernel = avg2d_kernel

        # Layers
        self.bn3d = nn.BatchNorm3d(self.channels, affine=True)
        self.adaptive_pool = nn.AdaptiveAvgPool3d(adaptive_out)
        self.avg2d = nn.AvgPool2d(kernel_size=self.avg2d_kernel, stride=self.avg2d_kernel)

        # Learnable mixing matrix for channel-wise feature interactions
        # We'll use it both for spatial mixing (applied to each spatial location) and final projection.
        self.mix = nn.Parameter(torch.randn(self.channels, self.channels) * (1.0 / (self.channels ** 0.5)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (N, C, D, H, W)

        Returns:
            Tensor of shape (N, C)
        """
        # 1. BatchNorm3d normalization
        x_norm = self.bn3d(x)  # (N, C, D, H, W)

        # 2. AdaptiveAvgPool3d to a small spatial grid
        pooled3d = self.adaptive_pool(x_norm)  # (N, C, 1, H', W')

        # 3. Squeeze depth dimension (we asserted D_out == 1)
        squeezed = pooled3d.squeeze(2)  # (N, C, H', W')

        # 4. 2D average pooling to reduce spatial resolution further
        pooled2d = self.avg2d(squeezed)  # (N, C, H2, W2)

        # 5. Flatten spatial dims and bring channels to last dim for mixing
        N, C, H2, W2 = pooled2d.shape
        spatial_flat = pooled2d.flatten(2)         # (N, C, S) where S = H2*W2
        spatial_feat = spatial_flat.permute(0, 2, 1)  # (N, S, C)

        # 6. Linear mixing across channels at each spatial location using self.mix
        mixed = torch.matmul(spatial_feat, self.mix)  # (N, S, C)

        # 7. Spatial aggregation (mean across spatial locations)
        aggregated = mixed.mean(dim=1)  # (N, C)

        # 8. Second projection: project aggregated features back through mixing matrix
        projected = torch.matmul(aggregated, self.mix)  # (N, C)

        # 9. Residual connection + nonlinearity
        out = torch.sigmoid(projected + aggregated)  # (N, C)

        return out

def get_inputs() -> List[torch.Tensor]:
    """
    Create a random 5D input tensor matching the configured shapes:
    (batch_size, channels, depth, height, width)
    """
    x = torch.randn(batch_size, channels, depth, height, width)
    return [x]

def get_init_inputs() -> List:
    """
    Return the inputs required to initialize the Model instance:
    - channels (int)
    - adaptive_output (tuple)
    - avg2d_kernel (int)
    """
    return [channels, adaptive_output, avgpool2d_kernel]