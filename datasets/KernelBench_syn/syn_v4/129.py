import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class Model(nn.Module):
    """
    A moderately complex 3D processing module that:
    - Applies a 3D max-pooling to compress spatial resolution
    - Projects per-voxel channel features into a hidden embedding (Linear)
    - Applies a non-linearity and projects back to channel space
    - Upsamples the processed tensor back to the original resolution and blends
      with a scaled residual connection from the input.
    This combines spatial pooling (nn.MaxPool3d) with channel-wise affine transforms (nn.Linear).
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        kernel_size: int = 2,
        stride: int = None,
        padding: int = 0,
        dilation: int = 1,
        negative_slope: float = 0.01,
        residual_scale: float = 0.1
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            hidden_dim (int): Hidden dimension for the channel-wise projection.
            kernel_size (int): MaxPool3d kernel size.
            stride (int, optional): MaxPool3d stride. If None, defaults to kernel_size.
            padding (int): MaxPool3d padding.
            dilation (int): MaxPool3d dilation.
            negative_slope (float): Negative slope for LeakyReLU.
            residual_scale (float): Scaling factor for the skip connection added to the output.
        """
        super(Model, self).__init__()
        if stride is None:
            stride = kernel_size

        # Spatial compressor
        self.pool = nn.MaxPool3d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)

        # Channel-wise projector: maps each voxel's channel vector -> hidden -> channel
        self.lin1 = nn.Linear(in_channels, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, in_channels)

        self.negative_slope = negative_slope
        self.residual_scale = residual_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        1. Save input for residual.
        2. Compress spatial dimensions with MaxPool3d.
        3. Reformat tensor so Linear operates over channels for each pooled-voxel.
        4. Apply linear projection -> activation -> linear projection.
        5. Restore pooled tensor layout.
        6. Upsample back to original spatial dimensions (trilinear) and add scaled residual.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, D, H, W)

        Returns:
            torch.Tensor: Output tensor of same shape as input.
        """
        identity = x  # (N, C, D, H, W)
        n, c, d, h, w = x.shape

        # 1) Spatial pooling
        pooled = self.pool(x)  # (N, C, d_p, h_p, w_p)

        # 2) Move channels to last dim and collapse spatial grid to sequence
        # pooled.permute -> (N, d_p, h_p, w_p, C)
        perm = pooled.permute(0, 2, 3, 4, 1)
        n, d_p, h_p, w_p, c = perm.shape
        seq_len = d_p * h_p * w_p
        flat = perm.reshape(n, seq_len, c)  # (N, S, C)

        # 3) Channel-wise projections using Linear layers (applied to last dim)
        hidden = self.lin1(flat)  # (N, S, hidden_dim)
        activated = F.leaky_relu(hidden, negative_slope=self.negative_slope)
        projected = self.lin2(activated)  # (N, S, C)

        # 4) Restore pooled spatial layout and channel-first convention
        restored = projected.reshape(n, d_p, h_p, w_p, c).permute(0, 4, 1, 2, 3)  # (N, C, d_p, h_p, w_p)

        # 5) Upsample back to original spatial resolution
        # Use trilinear interpolation to match (d, h, w)
        upsampled = F.interpolate(restored, size=(d, h, w), mode='trilinear', align_corners=False)

        # 6) Blend with a scaled residual connection to preserve low-level details
        out = upsampled + identity * self.residual_scale

        return out

# Module-level configuration
batch_size = 8
channels = 16
depth = 20
height = 32
width = 32

pool_kernel = 2
pool_stride = 2
pool_padding = 0
pool_dilation = 1

hidden_dim = 64
negative_slope = 0.02
residual_scale = 0.125

def get_inputs() -> List[torch.Tensor]:
    """
    Returns a list containing one input tensor consistent with the module-level configuration.
    Shape: (batch_size, channels, depth, height, width)
    """
    x = torch.randn(batch_size, channels, depth, height, width)
    return [x]

def get_init_inputs() -> List:
    """
    Returns the initialization inputs expected by Model.__init__ in order.
    """
    return [channels, hidden_dim, pool_kernel, pool_stride, pool_padding, pool_dilation, negative_slope, residual_scale]