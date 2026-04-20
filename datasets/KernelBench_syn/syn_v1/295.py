import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class Model(nn.Module):
    """
    Complex model that combines BatchNorm2d, MaxPool2d, RMSNorm, and a small
    channel-wise gating MLP. The computation flow is:
      1. BatchNorm2d over channels
      2. Spatial downsampling with MaxPool2d
      3. RMSNorm applied across the channel dimension for each spatial location
      4. Global spatial average to produce a channel descriptor
      5. Two-layer MLP (Linear -> SiLU -> Linear) to compute a sigmoid gating per channel
      6. Channel-wise gating applied and upsampling back to original spatial dimensions
      7. Residual add with the original input (if spatial sizes match)
    This pattern uses both channel-wise normalization (RMSNorm) and batch normalization,
    mixes local (per-pixel) and global (spatial average) information, and includes a small
    learned bottleneck for gating.
    """
    def __init__(self, in_channels: int, hidden_channels: int, pool_kernel: int = 2):
        """
        Args:
            in_channels (int): Number of input channels.
            hidden_channels (int): Hidden dimension for the channel-wise gating MLP.
            pool_kernel (int): Kernel (and stride) size for MaxPool2d downsampling.
        """
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.pool_kernel = pool_kernel

        # Normalize per-batch & channel for stable conv-like inputs
        self.bn = nn.BatchNorm2d(num_features=in_channels)

        # Spatial downsampling
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_kernel, ceil_mode=False)

        # RMSNorm will normalize across the channel dimension if we move channels to the last axis
        self.rms = nn.RMSNorm(normalized_shape=in_channels, eps=1e-8)

        # Small MLP to compute channel-wise gating from global spatial descriptors
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_channels, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W)
        """
        # 1. Batch normalization across the batch and spatial dims
        y = self.bn(x)  # (B, C, H, W)

        # 2. Spatial downsample to reduce computation for subsequent ops
        y = self.pool(y)  # (B, C, H', W')

        B, C, Hs, Ws = y.shape

        # 3. Move channels to last dim to apply RMSNorm over channels per spatial location
        #    y_perm shape: (B, H', W', C)
        y_perm = y.permute(0, 2, 3, 1)

        # 4. RMS normalization across the channel dimension for each (B, H', W')
        y_norm = self.rms(y_perm)  # (B, H', W', C)

        # 5. Move back to (B, C, H', W')
        y_norm = y_norm.permute(0, 3, 1, 2)

        # 6. Global spatial average to produce a channel descriptor
        g = y_norm.mean(dim=[2, 3])  # (B, C)

        # 7. Channel gating MLP: reduce -> nonlinearity -> expand -> sigmoid
        g = self.fc1(g)              # (B, hidden)
        g = self.act(g)
        g = self.fc2(g)              # (B, C)
        g = torch.sigmoid(g).view(B, C, 1, 1)  # (B, C, 1, 1)

        # 8. Apply gating to normalized features
        y_scaled = y_norm * g  # (B, C, H', W')

        # 9. Upsample back to original spatial size
        #    Use scale_factor equal to pool_kernel (assumed integer)
        out = F.interpolate(y_scaled, scale_factor=self.pool_kernel, mode='bilinear', align_corners=False)

        # 10. If the upsampled tensor matches the input spatial shape, add a residual connection
        if out.shape == x.shape:
            out = out + x

        return out

# Configuration variables for input generation
batch_size = 8
channels = 64
hidden_channels = 128
height = 128
width = 128
pool_kernel = 2

def get_inputs() -> List[torch.Tensor]:
    """
    Returns a list containing a single input tensor for inference/testing.

    Returns:
        List[torch.Tensor]: [x] where x has shape (batch_size, channels, height, width)
    """
    x = torch.randn(batch_size, channels, height, width)
    return [x]

def get_init_inputs() -> List:
    """
    Returns the initialization arguments for the Model constructor.

    Returns:
        List: [in_channels, hidden_channels, pool_kernel]
    """
    return [channels, hidden_channels, pool_kernel]