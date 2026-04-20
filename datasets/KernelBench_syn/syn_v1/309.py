import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

class Model(nn.Module):
    """
    Volumetric feature extractor that:
      - Applies Lp pooling in 3D to downsample the input volume.
      - Merges the depth and channel dimensions to treat depth as additional channels.
      - Uses Unfold to extract 2D sliding patches from the pooled feature maps.
      - Applies a per-patch linear projection followed by CELU nonlinearity.
      - Aggregates the patch features to produce a compact per-batch embedding.
    """
    def __init__(
        self,
        in_channels: int,
        in_depth: int,
        pool_p: int,
        pool_kernel: Tuple[int, int, int],
        unfold_kernel: Tuple[int, int],
        out_dim: int,
    ):
        """
        Args:
            in_channels: Number of input channels (C).
            in_depth: Depth dimension size (D).
            pool_p: The p-norm for LPPool3d.
            pool_kernel: Kernel size for LPPool3d (d, h, w).
            unfold_kernel: Kernel size for Unfold (h, w).
            out_dim: Output feature dimension per example.
        """
        super(Model, self).__init__()

        self.in_channels = in_channels
        self.in_depth = in_depth
        self.pool_p = pool_p
        self.pool_kernel = pool_kernel
        self.unfold_kernel = unfold_kernel
        self.out_dim = out_dim

        # 3D Lp pooling reduces (D, H, W) by the pool_kernel (we'll use stride == kernel to downsample)
        self.pool3d = nn.LPPool3d(self.pool_p, kernel_size=self.pool_kernel, stride=self.pool_kernel)

        # Unfold will be applied after merging depth into channel dimension, so its input channel
        # count will be computed at initialization to size the linear layer.
        # Compute pooled depth after LPPool3d
        pooled_d = self.in_depth // self.pool_kernel[0]
        if pooled_d < 1:
            raise ValueError("pool_kernel depth is too large for the provided in_depth.")

        # After pooling we will merge depth and channels: new_channels = in_channels * pooled_d
        new_channels = self.in_channels * pooled_d
        patch_h, patch_w = self.unfold_kernel
        patch_dim = new_channels * patch_h * patch_w

        # Unfold layer extracts patches from (N, new_channels, H', W')
        self.unfold = nn.Unfold(kernel_size=self.unfold_kernel)

        # Linear projection applied to each patch vector
        self.lin = nn.Linear(patch_dim, self.out_dim)

        # Non-linearity
        self.act = nn.CELU()

        # small normalization layer to stabilize final embedding (optional)
        self.norm_eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (N, C, D, H, W)

        Returns:
            Tensor of shape (N, out_dim) representing aggregated volumetric embeddings.
        """
        # Step 1: Lp pooling in 3D -> (N, C, D', H', W')
        pooled = self.pool3d(x)

        # Step 2: merge depth into the channel dimension so we can use 2D Unfold
        # pooled has shape (N, C, Dp, Hp, Wp)
        N, C, Dp, Hp, Wp = pooled.shape

        # Permute to (N, Dp, C, Hp, Wp) then reshape to (N, C*Dp, Hp, Wp)
        merged = pooled.permute(0, 2, 1, 3, 4).contiguous().view(N, C * Dp, Hp, Wp)

        # Step 3: extract 2D patches using Unfold -> (N, patch_dim, L)
        patches = self.unfold(merged)  # (N, patch_dim, L)

        # Step 4: rearrange to (N, L, patch_dim) for per-patch linear projection
        patches = patches.transpose(1, 2)  # (N, L, patch_dim)

        # Step 5: linear projection + CELU non-linearity
        projected = self.lin(patches)  # (N, L, out_dim)
        activated = self.act(projected)  # (N, L, out_dim)

        # Step 6: aggregate patch features (mean pooling across patches)
        aggregated = activated.mean(dim=1)  # (N, out_dim)

        # Optional: L2-normalize the embeddings for stability
        norm = aggregated.norm(p=2, dim=1, keepdim=True).clamp(min=self.norm_eps)
        output = aggregated / norm

        return output

# Configuration / constants
batch_size = 4
in_channels = 8
in_depth = 4
in_height = 32
in_width = 32

pool_p = 2  # L2 pooling
pool_kernel = (2, 2, 2)  # (d, h, w)
unfold_kernel = (3, 3)  # (h, w) for Unfold
out_dim = 128

def get_inputs() -> List[torch.Tensor]:
    """
    Returns a list with a single input tensor shaped (batch_size, in_channels, in_depth, in_height, in_width)
    filled with random values.
    """
    x = torch.randn(batch_size, in_channels, in_depth, in_height, in_width)
    return [x]

def get_init_inputs() -> List:
    """
    Returns the initialization inputs for the Model constructor in the same order as defined.
    """
    return [in_channels, in_depth, pool_p, pool_kernel, unfold_kernel, out_dim]