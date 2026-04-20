import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex 2D processing module that combines replication padding,
    Hardshrink non-linearity, local 3x3 neighborhood averaging (via unfold),
    Group Normalization, and a learnable channel-wise scaling with a residual connection.

    Computation steps:
    1. ReplicationPad2d to provide spatial context for border pixels.
    2. Hardshrink activation to sparsify/pulsate values.
    3. Extract 3x3 patches per channel with unfold and compute per-position local mean.
    4. Apply GroupNorm across channels.
    5. Multiply by a learnable per-channel scale parameter.
    6. Add the original input as a residual and apply a final Hardshrink.
    """
    def __init__(self, num_channels: int, num_groups: int = 8, pad: int = 1, hard_lambda: float = 0.3):
        """
        Initializes the module components.

        Args:
            num_channels (int): Number of channels in the input tensor.
            num_groups (int): Number of groups for GroupNorm.
            pad (int): Replication padding size applied to all sides.
            hard_lambda (float): Lambda threshold for Hardshrink.
        """
        super(Model, self).__init__()
        self.pad = pad
        # Replication padding to enlarge spatial context
        self.rep_pad = nn.ReplicationPad2d(pad)
        # Hardshrink used at two points to introduce sparsity/non-linearity
        self.hardshrink = nn.Hardshrink(lambd=hard_lambda)
        # Group normalization applied after local averaging
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
        # Learnable per-channel scaling parameter (broadcastable to BCHW)
        self.scale = nn.Parameter(torch.ones(1, num_channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the composed operations.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of the same shape (B, C, H, W).
        """
        B, C, H, W = x.shape

        # 1) Replication pad to provide valid 3x3 neighborhoods at borders
        x_padded = self.rep_pad(x)  # shape: (B, C, H + 2*pad, W + 2*pad)

        # 2) Initial hardshrink sparsification
        x_shrunk = self.hardshrink(x_padded)

        # 3) Extract 3x3 patches per channel and compute local mean per spatial location
        # F.unfold returns shape (B, C * kernel_area, L) where L = H_out * W_out = H * W (stride=1, no extra pad)
        patches = F.unfold(x_shrunk, kernel_size=3, padding=0, stride=1)  # (B, C*9, H*W)
        # Reshape to (B, C, 9, H*W) to average over the 3x3 neighborhood per channel
        patches = patches.reshape(B, C, 9, H * W)  # (B, C, 9, H*W)
        local_mean = patches.mean(dim=2)  # (B, C, H*W)
        # Reshape back to spatial map (B, C, H, W)
        local_mean = local_mean.view(B, C, H, W)

        # 4) Apply Group Normalization
        normalized = self.gn(local_mean)

        # 5) Channel-wise scaling (learnable) and residual connection
        scaled = normalized * self.scale  # broadcasting over spatial dims
        out = scaled + x  # residual connection to preserve original information

        # 6) Final hardshrink for additional sparsity
        out = self.hardshrink(out)

        return out

# Configuration / default inputs
batch_size = 8
channels = 48
height = 32
width = 32
pad = 1
num_groups = 8
hard_lambda = 0.3

def get_inputs():
    """
    Returns a list with a single input tensor shaped (batch_size, channels, height, width).
    """
    x = torch.randn(batch_size, channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for the Model:
    [num_channels, num_groups, pad, hard_lambda]
    """
    return [channels, num_groups, pad, hard_lambda]