import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

# Module-level configuration
batch_size = 8
channels = 16
depth = 20
height = 24
width = 24

# Adaptive pooling target output spatial size (D_out, H_out, W_out)
pool_output_size = (4, 3, 3)

# Output feature size for the Bilinear layer
bilinear_out_features = 128


class Model(nn.Module):
    """
    Complex 3D-feature interaction model that:
    - Pools two 3D feature maps to a fixed spatial grid using AdaptiveMaxPool3d
    - Applies LazyInstanceNorm3d to each pooled feature map (lazy initialization for channels)
    - Flattens pooled spatial dimensions and computes a bilinear interaction between the two
      flattened representations to produce a compact joint descriptor per batch element
    - Applies a non-linear activation to the bilinear outputs

    Inputs:
        x1, x2: torch.Tensor of shape (batch_size, channels, D, H, W)

    Outputs:
        torch.Tensor of shape (batch_size, bilinear_out_features)
    """

    def __init__(self, pool_size: Tuple[int, int, int], out_features: int):
        """
        Args:
            pool_size (Tuple[int, int, int]): Target output size for AdaptiveMaxPool3d (D_out, H_out, W_out)
            out_features (int): Number of output features from the bilinear interaction
        """
        super(Model, self).__init__()

        self.pool_size = tuple(pool_size)
        self.out_features = int(out_features)

        # Adaptive spatial pooling to reduce arbitrary input volumes to a fixed-size grid
        self.adaptive_pool = nn.AdaptiveMaxPool3d(output_size=self.pool_size)

        # LazyInstanceNorm3d will infer num_features from the incoming tensor's channel dim on first forward
        self.instnorm1 = nn.LazyInstanceNorm3d()
        self.instnorm2 = nn.LazyInstanceNorm3d()

        # Compute the number of features produced after pooling and flattening:
        # in_features = channels * (D_out * H_out * W_out)
        pooled_spatial_size = self.pool_size[0] * self.pool_size[1] * self.pool_size[2]
        in_features = channels * pooled_spatial_size

        # Bilinear layer to learn a multiplicative interaction between the two flattened descriptors
        self.bilinear = nn.Bilinear(in1_features=in_features, in2_features=in_features, out_features=self.out_features, bias=True)

        # Small projection + activation after bilinear to increase expressivity
        self.post_proj = nn.Sequential(
            nn.Linear(self.out_features, self.out_features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing pooled, normalized, and bilinearly combined descriptors.

        Steps:
        1. Adaptive max pool both inputs to a fixed grid.
        2. Apply LazyInstanceNorm3d independently to each pooled tensor.
        3. Flatten spatial dimensions and channels to a single feature vector per example.
        4. Compute bilinear interaction between the two vectors per example.
        5. Apply a small projection + ReLU and return the result.

        Args:
            x1 (torch.Tensor): Tensor of shape (B, C, D, H, W)
            x2 (torch.Tensor): Tensor of shape (B, C, D, H, W)

        Returns:
            torch.Tensor: Tensor of shape (B, out_features)
        """
        # 1) Pooling
        p1 = self.adaptive_pool(x1)  # shape: (B, C, D_out, H_out, W_out)
        p2 = self.adaptive_pool(x2)

        # 2) Instance normalization (lazy so num_features inferred)
        n1 = self.instnorm1(p1)
        n2 = self.instnorm2(p2)

        # 3) Flatten spatial dims and channels -> (B, C * D_out * H_out * W_out)
        B = n1.size(0)
        flattened1 = n1.view(B, -1)
        flattened2 = n2.view(B, -1)

        # 4) Bilinear interaction per example -> (B, out_features)
        b_out = self.bilinear(flattened1, flattened2)

        # 5) Small projection and non-linearity
        out = self.post_proj(b_out)

        return out


def get_inputs() -> List[torch.Tensor]:
    """
    Creates a pair of random 3D feature map tensors to feed the model.

    Returns:
        List[torch.Tensor]: [x1, x2] each of shape (batch_size, channels, depth, height, width)
    """
    x1 = torch.randn(batch_size, channels, depth, height, width)
    x2 = torch.randn(batch_size, channels, depth, height, width)
    return [x1, x2]


def get_init_inputs() -> List:
    """
    Returns initialization parameters for the Model constructor.

    Returns:
        List: [pool_output_size, bilinear_out_features]
    """
    return [pool_output_size, bilinear_out_features]