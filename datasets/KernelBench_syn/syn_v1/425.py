import torch
import torch.nn as nn

# Configuration variables
batch_size = 8
channels = 64
depth = 16
height = 128
width = 128
pooled_h = 8
pooled_w = 8
dropout_p = 0.2

class Model(nn.Module):
    """
    Volumetric feature processor that:
    - Applies 3D dropout over channels
    - Pools each depth-slice spatially with AdaptiveAvgPool2d
    - Normalizes via LazyBatchNorm2d (lazy channel init)
    - Computes a depth-wise gating by mixing per-depth statistics with a learnable depth mixing matrix
    - Applies the gating to the pooled feature maps and collapses the depth dimension

    Input shape: (N, C, D, H, W)
    Output shape: (N, C, pooled_h, pooled_w)
    """
    def __init__(self):
        super(Model, self).__init__()
        # Drop entire channels across the volumetric input
        self.drop3d = nn.Dropout3d(p=dropout_p)

        # Spatial pooling for each depth-slice (we will reshape to (N*D, C, H, W) before applying)
        self.pool2d = nn.AdaptiveAvgPool2d((pooled_h, pooled_w))

        # Lazy batchnorm over channel dimension; will initialize when the first forward is run
        self.bn = nn.LazyBatchNorm2d()

        # Learnable depth mixing matrix to model interactions across depth positions
        # Initialized with small random values; shape = (D, D)
        self.depth_mix = nn.Parameter(torch.randn(depth, depth) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Volumetric input tensor of shape (N, C, D, H, W).

        Returns:
            torch.Tensor: Processed tensor of shape (N, C, pooled_h, pooled_w).
        """
        N, C, D, H, W = x.shape
        assert C == channels and D == depth and H == height and W == width, \
            f"Expected input shape (N, {channels}, {depth}, {height}, {width}), got {x.shape}"

        # 1) Randomly drop entire channels
        x = self.drop3d(x)  # (N, C, D, H, W)

        # 2) Bring depth into batch dimension to process each depth-slice with 2D ops
        x_perm = x.permute(0, 2, 1, 3, 4).contiguous()  # (N, D, C, H, W)
        x_reshaped = x_perm.view(N * D, C, H, W)        # (N*D, C, H, W)

        # 3) Spatial pooling per depth-slice
        pooled = self.pool2d(x_reshaped)                # (N*D, C, pooled_h, pooled_w)

        # 4) Channel normalization (LazyBatchNorm2d will infer C on first call)
        pooled_bn = self.bn(pooled)                      # (N*D, C, pooled_h, pooled_w)

        # 5) Compute per-depth, per-channel summary (global spatial average of pooled maps)
        spatial_avg = pooled_bn.mean(dim=[2, 3])        # (N*D, C)
        spatial_avg = spatial_avg.view(N, D, C)        # (N, D, C)

        # 6) Permute to (N, C, D) to mix across depth for each channel independently
        avg_nc_d = spatial_avg.permute(0, 2, 1)        # (N, C, D)

        # 7) Depth mixing: apply learnable depth mixing matrix to model interactions across depth
        # mixed: (N, C, D) = (N, C, D) @ (D, D)
        mixed = torch.matmul(avg_nc_d, self.depth_mix)  # (N, C, D)

        # 8) Compute gating and apply to the pooled feature maps
        gating = torch.sigmoid(mixed)                   # (N, C, D)
        gating_cd = gating.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # (N, D, C, 1, 1)

        pooled_bn_reshaped = pooled_bn.view(N, D, C, pooled_h, pooled_w)  # (N, D, C, P, Q)
        gated_features = pooled_bn_reshaped * gating_cd                   # (N, D, C, P, Q)

        # 9) Collapse depth dimension by averaging (could also max or learnable reduction)
        out = gated_features.mean(dim=1)  # (N, C, pooled_h, pooled_w)

        return out

def get_inputs():
    """
    Returns a list containing a single input tensor shaped (batch_size, channels, depth, height, width).
    """
    x = torch.randn(batch_size, channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    No external initialization parameters required for this module.
    """
    return []