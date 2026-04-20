import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A composite 2D feature mixer that combines BatchNorm2d, channel-wise learned
    linear mixing (implemented with nn.Parameter + matmul), GroupNorm and ReLU,
    with a learned residual channel projection.

    The model workflow:
      1. BatchNorm2d + ReLU on input feature maps.
      2. Reshape to (B, H*W, C) and perform two learned channel projections
         (C -> mid -> out) using torch.matmul to mix channel information
         at each spatial location.
      3. Reshape back to (B, out, H, W).
      4. Apply GroupNorm over the output channels.
      5. Add a residual path computed by projecting the original input channels
         to out channels (per spatial location) using a learned skip projection.
      6. Final ReLU activation.

    This creates a spatially-preserving, fully-learned channel mixing block
    that is functionally distinct from simple convs or pure normalization layers.
    """
    def __init__(self,
                 in_channels: int,
                 mid_channels: int,
                 out_channels: int,
                 bn_momentum: float = 0.1,
                 gn_num_groups: int = 4):
        """
        Initialize the module.

        Args:
            in_channels (int): Number of channels in the input tensor.
            mid_channels (int): Intermediate channel dimension for channel mixing.
            out_channels (int): Number of output channels after mixing.
            bn_momentum (float): Momentum for BatchNorm2d.
            gn_num_groups (int): Number of groups for GroupNorm.
        """
        super(Model, self).__init__()
        # BatchNorm operates on the original input channels
        self.bn = nn.BatchNorm2d(in_channels, momentum=bn_momentum)
        # ReLU activation
        self.relu = nn.ReLU(inplace=True)
        # Learnable channel mixing matrices (implemented as Parameters)
        # First projection: in_channels -> mid_channels
        self.proj1 = nn.Parameter(torch.randn(in_channels, mid_channels) * (1.0 / max(1, in_channels)**0.5))
        # Second projection: mid_channels -> out_channels
        self.proj2 = nn.Parameter(torch.randn(mid_channels, out_channels) * (1.0 / max(1, mid_channels)**0.5))
        # Residual channel projection: in_channels -> out_channels
        self.skip_proj = nn.Parameter(torch.randn(in_channels, out_channels) * (1.0 / max(1, in_channels)**0.5))
        # GroupNorm applied after channel mixing
        self.gn = nn.GroupNorm(num_groups=gn_num_groups, num_channels=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H, W).
        """
        B, C_in, H, W = x.shape
        # 1) Batch normalization and non-linearity
        y = self.bn(x)                # (B, C_in, H, W)
        y = self.relu(y)              # (B, C_in, H, W)

        # 2) Prepare for channel mixing: move channels to last dimension per spatial location
        #    shape -> (B, N, C_in) where N = H*W
        N = H * W
        y_perm = y.permute(0, 2, 3, 1).reshape(B, N, C_in)

        # 3) Channel mixing stage 1: (B, N, C_in) @ (C_in, mid) -> (B, N, mid)
        y_mixed = torch.matmul(y_perm, self.proj1)

        # 4) Non-linearity in the mid-space
        y_mixed = self.relu(y_mixed)

        # 5) Channel mixing stage 2: (B, N, mid) @ (mid, out) -> (B, N, C_out)
        y_out = torch.matmul(y_mixed, self.proj2)

        # 6) Reshape back to (B, C_out, H, W)
        y_out = y_out.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        # 7) Group normalization over output channels
        y_out = self.gn(y_out)

        # 8) Residual path: project original input channels to out channels per spatial location
        #    Compute skip by reshaping input similarly and matmul with skip_proj
        skip_perm = x.permute(0, 2, 3, 1).reshape(B, N, C_in)  # (B, N, C_in)
        skip_mapped = torch.matmul(skip_perm, self.skip_proj)  # (B, N, C_out)
        skip = skip_mapped.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # (B, C_out, H, W)

        # 9) Combine and final activation
        out = self.relu(y_out + skip)

        return out

# Module-level configuration variables
batch_size = 8
in_channels = 64
mid_channels = 128
out_channels = 96
gn_num_groups = 8
height = 32
width = 32

def get_inputs():
    """
    Returns a list with a single input tensor of shape (batch_size, in_channels, height, width).
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters for the Model constructor:
    [in_channels, mid_channels, out_channels, bn_momentum, gn_num_groups]
    """
    return [in_channels, mid_channels, out_channels, 0.1, gn_num_groups]