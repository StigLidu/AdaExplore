import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class Model(nn.Module):
    """
    Complex 3D feature processing module that combines Group Normalization,
    Tanh activation, MaxPool3d, and a small squeeze-and-excitation style
    channel reweighting using Linear layers. The model demonstrates a multi-step
    pipeline with spatial pooling, channel attention, and a final normalization.
    """
    def __init__(self, in_channels: int, num_groups: int, pool_kernel: int = 2, reduction: int = 4):
        """
        Initializes the composite module.

        Args:
            in_channels (int): Number of input channels (C) for the 5D input (N, C, D, H, W).
            num_groups (int): Preferred number of groups for GroupNorm. The actual
                              groups used will be adjusted so that num_channels % groups == 0.
            pool_kernel (int): Kernel (and stride) size for MaxPool3d.
            reduction (int): Reduction factor for the squeeze bottleneck (must be >=1).
        """
        super(Model, self).__init__()

        assert in_channels > 0, "in_channels must be positive"
        assert pool_kernel >= 1, "pool_kernel must be >= 1"
        reduction = max(1, reduction)
        hidden_channels = max(1, in_channels // reduction)

        # Adjust groups for the input GroupNorm so that in_channels % groups == 0
        groups_in = min(num_groups, in_channels)
        while groups_in > 1 and (in_channels % groups_in) != 0:
            groups_in -= 1
        # At worst, groups_in == 1 (LayerNorm behavior)
        if groups_in < 1:
            groups_in = 1

        # After a later concatenation we will have (in_channels + 1) channels; compute groups for output GN
        out_channels = in_channels + 1
        groups_out = min(num_groups, out_channels)
        while groups_out > 1 and (out_channels % groups_out) != 0:
            groups_out -= 1
        if groups_out < 1:
            groups_out = 1

        # Layers and components
        self.gn_in = nn.GroupNorm(num_groups=groups_in, num_channels=in_channels)
        self.activation = nn.Tanh()
        self.pool = nn.MaxPool3d(kernel_size=pool_kernel, stride=pool_kernel, padding=0, ceil_mode=False)

        # Small MLP for channel-wise gating (squeeze-and-excite style)
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, in_channels)

        # Final normalization after concatenation of spatial max map
        self.gn_out = nn.GroupNorm(num_groups=groups_out, num_channels=out_channels)

        # Save configuration
        self.in_channels = in_channels
        self.pool_kernel = pool_kernel
        self.reduction = reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the composite module.

        Steps:
        1. Group Normalization on channels.
        2. Tanh activation.
        3. 3D Max Pooling to reduce spatial resolution.
        4. Global average pooling (spatial mean) to produce channel descriptors.
        5. Small two-layer MLP with Tanh non-linearity to compute channel weights.
        6. Scale pooled feature maps by channel weights.
        7. Compute spatial max across channels and concatenate as an extra channel.
        8. Apply final Group Normalization and return.

        Args:
            x (torch.Tensor): Input tensor with shape (N, C, D, H, W).

        Returns:
            torch.Tensor: Output tensor with shape (N, C+1, D', H', W') where
                          D', H', W' are downsampled by pool_kernel.
        """
        # Input normalization and activation
        y = self.gn_in(x)
        y = self.activation(y)

        # Spatial downsampling
        y = self.pool(y)

        # Channel squeeze: global spatial average -> (N, C)
        # mean over dims 2,3,4 (D, H, W)
        gap = y.mean(dim=(2, 3, 4))  # shape (N, C)

        # Channel MLP: (N, C) -> (N, hidden) -> (N, C)
        z = self.fc1(gap)
        z = self.activation(z)
        z = self.fc2(z)
        # Map tanh outputs from [-1,1] to [0,1] for gating
        weights = (torch.tanh(z) + 1.0) * 0.5  # shape (N, C)

        # Reshape for broadcasting and apply channel scaling
        weights = weights.view(weights.size(0), weights.size(1), 1, 1, 1)  # (N, C, 1, 1, 1)
        y_scaled = y * weights

        # Spatial summary: max across channels -> (N, 1, D', H', W')
        spatial_max = y_scaled.max(dim=1, keepdim=True)[0]

        # Concatenate the scaled feature maps with the spatial max channel
        out = torch.cat([y_scaled, spatial_max], dim=1)  # (N, C+1, D', H', W')

        # Final normalization to mix channels
        out = self.gn_out(out)

        return out


# -----------------------
# Module-level configuration
# -----------------------
batch_size = 4
in_channels = 32
num_groups = 8
depth = 16
height = 32
width = 32
pool_kernel = 2
reduction = 4

def get_inputs() -> List[torch.Tensor]:
    """
    Returns a list with the main input tensor for the model:
    A random tensor shaped (batch_size, in_channels, depth, height, width).
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs() -> List:
    """
    Returns the initialization parameters for the Model class:
    [in_channels, num_groups, pool_kernel, reduction]
    """
    return [in_channels, num_groups, pool_kernel, reduction]