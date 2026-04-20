import torch
import torch.nn as nn
from typing import List, Tuple

class Model(nn.Module):
    """
    Complex model that combines 3D Lp-pooling, 2D unfolding of spatial patches per depth-slice,
    channel-wise Dropout1d over extracted patch-features, a learned linear projection per patch,
    and aggregation across spatial patches and depth slices.

    Flow:
        x (N, C, D, H, W)
        -> LPPool3d -> (N, C, Dp, Hp, Wp)
        -> reshape to (N*Dp, C, Hp, Wp)
        -> Unfold -> (N*Dp, C * k * k, L)  where k = unfold_kernel
        -> Dropout1d over the channel-like dimension (C * k * k)
        -> transpose -> (N*Dp, L, feat) and apply nn.Linear to map feat -> proj_dim
        -> mean over L (spatial positions) -> (N*Dp, proj_dim)
        -> reshape to (N, Dp, proj_dim) and reduce over depth via max -> (N, proj_dim)
    """
    def __init__(
        self,
        channels: int,
        lp_norm: int,
        lp_kernel: Tuple[int, int, int],
        unfold_kernel: int,
        unfold_stride: int,
        dropout_p: float,
        proj_dim: int,
        unfold_padding: int = 0
    ):
        """
        Initializes the module.

        Args:
            channels (int): Number of input channels C.
            lp_norm (int): The p value for LPPool3d (norm type).
            lp_kernel (tuple): Kernel size for LPPool3d as (kd, kh, kw).
            unfold_kernel (int): Kernel size for nn.Unfold over (H, W).
            unfold_stride (int): Stride for nn.Unfold.
            dropout_p (float): Dropout probability for Dropout1d.
            proj_dim (int): Output dimensionality of the per-patch linear projection.
            unfold_padding (int): Padding for unfold operation (default 0).
        """
        super(Model, self).__init__()
        # 3D Lp-pooling layer
        self.lp_pool = nn.LPPool3d(norm_type=lp_norm, kernel_size=lp_kernel)

        # Unfold for sliding window extraction over 2D spatial dims (H, W)
        self.unfold = nn.Unfold(kernel_size=unfold_kernel, stride=unfold_stride, padding=unfold_padding)

        # Dropout across the "channel" dimension of extracted patches (C * k * k)
        self.dropout = nn.Dropout1d(p=dropout_p)

        # Linear projection applied per extracted patch feature vector
        in_feat = channels * (unfold_kernel * unfold_kernel)
        self.proj = nn.Linear(in_features=in_feat, out_features=proj_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, D, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (N, proj_dim)
        """
        # Apply 3D Lp-pooling: (N, C, Dp, Hp, Wp)
        pooled = self.lp_pool(x)

        N, C, Dp, Hp, Wp = pooled.shape

        # Collapse batch and depth to treat each depth-slice as an independent image:
        # (N * Dp, C, Hp, Wp)
        pooled_slices = pooled.permute(0, 2, 1, 3, 4).contiguous().view(N * Dp, C, Hp, Wp)

        # Extract sliding local blocks: (N*Dp, C * k * k, L)
        unfolded = self.unfold(pooled_slices)

        # Apply Dropout1d across the channel-like dim (C * k * k)
        dropped = self.dropout(unfolded)

        # Transpose to shape (N*Dp, L, feat) to project each patch-vector
        patches = dropped.permute(0, 2, 1)  # (N*Dp, L, feat)

        # Linear projection applied to the last dimension: (..., feat) -> (..., proj_dim)
        projected = self.proj(patches)  # (N*Dp, L, proj_dim)

        # Aggregate across spatial locations (L) by mean -> (N*Dp, proj_dim)
        spatial_agg = projected.mean(dim=1)

        # Restore depth dimension: (N, Dp, proj_dim)
        depth_view = spatial_agg.view(N, Dp, -1)

        # Final aggregation across depth slices: take max across depth -> (N, proj_dim)
        final = depth_view.max(dim=1).values

        return final


# Module-level configuration variables (example settings)
batch_size = 4
channels = 8
depth = 6
height = 32
width = 32

lp_norm = 2  # p for LPPool3d
lp_kernel = (2, 2, 2)  # reduces D,H,W roughly by factor of 2
unfold_kernel = 3
unfold_stride = 2
unfold_padding = 0
dropout_p = 0.2
proj_dim = 64

def get_inputs() -> List[torch.Tensor]:
    """
    Returns example input tensors for the model.
    Input shape: (batch_size, channels, depth, height, width)
    """
    x = torch.randn(batch_size, channels, depth, height, width)
    return [x]

def get_init_inputs() -> List:
    """
    Returns the initialization parameters for the Model class in the same order
    as the constructor arguments (excluding any defaults after those provided).
    """
    return [
        channels,
        lp_norm,
        lp_kernel,
        unfold_kernel,
        unfold_stride,
        dropout_p,
        proj_dim,
        unfold_padding,
    ]