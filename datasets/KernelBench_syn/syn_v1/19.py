import torch
import torch.nn as nn

"""
Complex 3D processing module that demonstrates a combination of:
- LazyBatchNorm3d for lazy channel-wise normalization
- LPPool3d for Lp pooling in 3D (D, H, W)
- Dropout2d applied cleverly across depth slices by reshaping
The model performs:
    input -> LazyBatchNorm3d -> LPPool3d -> Dropout2d (applied per depth-slice) -> ReLU
    -> Global spatial average (over D,H,W) -> channel-wise scaling using BatchNorm weight (if available)
Outputs a (batch_size, channels) tensor (per-channel pooled descriptors).
"""

# Configuration variables
batch_size = 8
channels = 12       # number of channels (will be discovered lazily by LazyBatchNorm3d)
depth = 6
height = 32
width = 32

lp_norm = 2         # p for LPPool3d
pool_kernel = (2, 2, 2)
pool_stride = pool_kernel
dropout_prob = 0.25

class Model(nn.Module):
    """
    A Model that composes LazyBatchNorm3d, LPPool3d and Dropout2d in a non-trivial pattern.
    It normalizes the input across channels (lazy), reduces spatial resolution with LPPool3d,
    applies channel dropout per depth-slice using Dropout2d (by reshaping), then aggregates
    spatially and scales channel descriptors using the BatchNorm weight (if present).
    """
    def __init__(self,
                 lp_norm: int = lp_norm,
                 pool_kernel=(2, 2, 2),
                 pool_stride=None,
                 dropout_p: float = dropout_prob):
        super(Model, self).__init__()
        # Lazy batch norm will infer num_features on first forward pass
        self.bn3d = nn.LazyBatchNorm3d()
        # LPPool3d: norm_type, kernel_size, stride
        self.pool = nn.LPPool3d(norm_type=lp_norm, kernel_size=pool_kernel, stride=pool_stride)
        # Dropout2d will be applied to 4D tensors; we will reshape the 5D tensor to apply it per depth slice
        self.dropout2d = nn.Dropout2d(p=dropout_p)
        # Small epsilon for numerical stability during global pooling
        self.eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation:
        - x: (N, C, D, H, W)
        Returns:
        - Tensor of shape (N, C) containing per-channel descriptors after pooling & scaling.
        """
        # Validate input dims
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input (N, C, D, H, W), got shape {tuple(x.shape)}")

        N, C, D, H, W = x.shape

        # 1) Channel-wise normalization (lazy)
        x = self.bn3d(x)  # still (N, C, D, H, W)

        # 2) Lp pooling to reduce spatial dimensions (D, H, W)
        x = self.pool(x)  # -> (N, C, D2, H2, W2)
        N, C, D2, H2, W2 = x.shape

        # 3) Apply Dropout2d across channels per depth-slice.
        # Dropout2d expects 4D: (batch_slice, C, H2, W2). We treat each depth slice as a separate batch.
        # Permute to bring depth next to batch, reshape, apply dropout, then restore shape.
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (N, D2, C, H2, W2)
        x = x.view(N * D2, C, H2, W2)              # (N*D2, C, H2, W2)
        x = self.dropout2d(x)                      # dropout applied per (C, H2, W2)
        x = x.view(N, D2, C, H2, W2)               # (N, D2, C, H2, W2)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # back to (N, C, D2, H2, W2)

        # 4) Non-linearity
        x = torch.relu(x)

        # 5) Global spatial average pooling over (D2, H2, W2) to produce (N, C)
        # add small eps to protect against empty tensors in pathological cases
        x = x.mean(dim=(2, 3, 4))  # (N, C)

        # 6) Channel-wise scaling using BatchNorm weight if available (after lazy init bn3d)
        if hasattr(self.bn3d, 'weight') and self.bn3d.weight is not None:
            # bn3d.weight is of shape (C,), reshape to (1, C) to broadcast over batch
            x = x * (self.bn3d.weight.view(1, -1) + self.eps)

        return x


def get_inputs():
    """
    Generate a random input tensor of shape (batch_size, channels, depth, height, width)
    The channel dimension will be discovered by LazyBatchNorm3d on the first forward pass.
    """
    x = torch.randn(batch_size, channels, depth, height, width)
    return [x]


def get_init_inputs():
    """
    No special initialization inputs needed because LazyBatchNorm3d initializes lazily.
    """
    return []