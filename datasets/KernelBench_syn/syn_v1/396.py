import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex model that demonstrates mixing 3D batch normalization, 2D Lp-pooling applied across
    depth slices, 1D replication padding along the width axis, and a channel projection via a linear layer.

    Computation steps:
      1. BatchNorm3d across (B, C, D, H, W)
      2. For each depth slice, treat slice as a 4D tensor (B*D, C, H, W) and apply LPPool2d
      3. For each pooled row, apply ReplicationPad1d along the width dimension, then reduce (mean) across width
      4. Project channel features with a Linear layer to a smaller channel dimension
      5. Reassemble to (B, C_out, D, H_pooled) and global-average across (D, H_pooled) to produce (B, C_out)
    """
    def __init__(
        self,
        in_channels: int,
        pool_p: int,
        pool_kernel: int,
        pool_stride: int,
        pad_left: int,
        pad_right: int,
        out_channels: int,
    ):
        super(Model, self).__init__()
        # Normalize across (B, C, D, H, W)
        self.bn3d = nn.BatchNorm3d(in_channels)
        # Lp pooling applied to each depth slice (operates on 4D tensors: N, C, H, W)
        self.lppool = nn.LPPool2d(norm_type=pool_p, kernel_size=pool_kernel, stride=pool_stride)
        # Replication pad for 1D sequences (operates on 3D tensors: N, C, L)
        self.pad1d = nn.ReplicationPad1d((pad_left, pad_right))
        # Linear projection from in_channels -> out_channels for each spatial-temporal position
        self.fc = nn.Linear(in_channels, out_channels)
        # Keep attributes for reshaping logic
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pool_stride = pool_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, D, H, W)

        Returns:
            Tensor of shape (B, out_channels) after spatial-temporal aggregation
        """
        B, C, D, H, W = x.shape  # Expected shape
        # 1) BatchNorm3d
        x = self.bn3d(x)  # (B, C, D, H, W)

        # 2) Merge batch and depth to apply 2D pooling per depth slice
        # Rearrange to (B, D, C, H, W) -> (B*D, C, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * D, C, H, W)  # (B*D, C, H, W)

        # Apply LPPool2d across spatial dims (H, W)
        x = self.lppool(x)  # (B*D, C, H_p, W_p)
        _, _, H_p, W_p = x.shape

        # 3) Prepare for ReplicationPad1d: treat each row (height) as a separate sequence along width
        # Current x shape: (B*D, C, H_p, W_p)
        # Permute to (B*D, H_p, C, W_p) then merge first two dims to get (B*D*H_p, C, W_p)
        x = x.permute(0, 2, 1, 3).contiguous().view(B * D * H_p, C, W_p)  # (B*D*H_p, C, W_p)

        # Apply 1D replication padding along width dimension
        x = self.pad1d(x)  # (B*D*H_p, C, W_p_padded)

        # 4) Reduce across the width dimension (e.g., average pooling along the sequence)
        x = x.mean(dim=2)  # (B*D*H_p, C)

        # 5) Channel projection via Linear layer
        x = self.fc(x)  # (B*D*H_p, out_channels)

        # 6) Reshape back to (B, out_channels, D, H_p)
        x = x.view(B, D, H_p, self.out_channels).permute(0, 3, 1, 2).contiguous()  # (B, out_channels, D, H_p)

        # 7) Global average across depth and pooled height -> produce per-batch feature vector
        out = x.mean(dim=(2, 3))  # (B, out_channels)

        return out


# Configuration variables
BATCH = 8
IN_CHANNELS = 32
DEPTH = 16
HEIGHT = 64
WIDTH = 64

POOL_P = 2
POOL_KERNEL = 3
POOL_STRIDE = 2

PAD_LEFT = 1
PAD_RIGHT = 2

OUT_CHANNELS = 16

def get_inputs():
    """
    Returns:
        [x] where x is a random input tensor of shape (BATCH, IN_CHANNELS, DEPTH, HEIGHT, WIDTH)
    """
    x = torch.randn(BATCH, IN_CHANNELS, DEPTH, HEIGHT, WIDTH)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for Model constructor in the same order:
      (in_channels, pool_p, pool_kernel, pool_stride, pad_left, pad_right, out_channels)
    """
    return [IN_CHANNELS, POOL_P, POOL_KERNEL, POOL_STRIDE, PAD_LEFT, PAD_RIGHT, OUT_CHANNELS]