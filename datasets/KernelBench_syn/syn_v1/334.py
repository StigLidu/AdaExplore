import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

"""
Complex PyTorch module combining:
- nn.Conv2d (1x1 channel expansion)
- nn.LazyBatchNorm2d (lazy-initialized batch normalization)
- nn.AdaptiveAvgPool2d (spatial pooling)
- nn.Softsign (non-linearity)
Plus channel-wise gating, channel shuffle, residual connection, and spatial interpolation.

Structure follows the provided examples:
- Model class inheriting from nn.Module
- get_inputs() for runtime tensors
- get_init_inputs() for constructor parameters
- module-level configuration variables
"""

# Configuration variables (module-level)
batch_size = 8
in_channels = 3
expand_channels = 32   # must be divisible by channel_groups
height = 128
width = 64
pool_output_size = (4, 4)  # target size for first adaptive pooling
final_pool_output = (1, 1)  # target size for final adaptive pooling
bn_eps = 1e-5
channel_groups = 4  # groups for channel shuffle (must divide expand_channels)

class Model(nn.Module):
    """
    Complex 2D feature transformer that:
    1) Expands channels with a 1x1 convolution
    2) Applies lazy BatchNorm2d (initialized on first forward)
    3) Uses an AdaptiveAvgPool2d to create a channel descriptor
    4) Applies Softsign to descriptor and gates feature maps
    5) Performs a channel shuffle to mix group-wise channels
    6) Adds a residual path (projected) and applies final adaptive pooling
    """
    def __init__(self,
                 in_channels: int,
                 expand_channels: int,
                 pool_output_size: Tuple[int, int],
                 final_pool_output: Tuple[int, int],
                 bn_eps: float = 1e-5,
                 channel_groups: int = 4):
        """
        Args:
            in_channels (int): Number of channels in the input tensor.
            expand_channels (int): Number of channels after 1x1 expansion.
            pool_output_size (Tuple[int,int]): Output spatial size for the intermediate adaptive pooling.
            final_pool_output (Tuple[int,int]): Output spatial size for the final adaptive pooling.
            bn_eps (float): Epsilon value for BatchNorm.
            channel_groups (int): Number of groups for channel shuffle (must divide expand_channels).
        """
        super(Model, self).__init__()
        if expand_channels % channel_groups != 0:
            raise ValueError("expand_channels must be divisible by channel_groups")

        # 1x1 conv for channel expansion
        self.conv_expand = nn.Conv2d(in_channels, expand_channels, kernel_size=1, bias=True)

        # Lazy batchnorm: will be initialized when conv_expand's output is seen
        self.bn = nn.LazyBatchNorm2d(eps=bn_eps)

        # Intermediate adaptive pooling to build channel descriptor
        self.adaptive_pool = nn.AdaptiveAvgPool2d(pool_output_size)

        # Softsign non-linearity for gating
        self.softsign = nn.Softsign()

        # Channel groups for shuffle
        self.channel_groups = channel_groups
        self.expand_channels = expand_channels

        # Small projection for skip/residual path to match expanded channels
        self.skip_proj = nn.Conv2d(in_channels, expand_channels, kernel_size=1, bias=True)

        # Final adaptive pooling to produce compact outputs
        self.final_pool = nn.AdaptiveAvgPool2d(final_pool_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        - x: (B, C_in, H, W)
        Returns:
        - Tensor of shape (B, expand_channels, final_pool_h, final_pool_w)
        """
        # Expand channels
        z = self.conv_expand(x)                      # (B, C_exp, H, W)

        # Normalize (lazy initialization of bn happens here)
        z = self.bn(z)                               # (B, C_exp, H, W)

        # Non-linear activation (using built-in ReLU for internal non-linearity)
        z = F.relu(z, inplace=False)                 # (B, C_exp, H, W)

        # Build channel descriptor via adaptive pooling then spatial average
        pooled = self.adaptive_pool(z)               # (B, C_exp, pH, pW)
        descriptor = pooled.mean(dim=(-2, -1), keepdim=True)  # (B, C_exp, 1, 1)

        # Gate using Softsign: values in (-1, 1), shift to positive gating multiplier
        gate = self.softsign(descriptor)             # (B, C_exp, 1, 1)
        gated = z * (1.0 + gate)                     # broadcast multiply (residual-style gating)

        # Channel shuffle to encourage cross-group mixing
        B, C, H, W = gated.shape
        g = self.channel_groups
        # Reshape to (B, g, C/g, H, W), transpose group and channel_in_group, then flatten
        gated = gated.view(B, g, C // g, H, W)
        gated = gated.transpose(1, 2).contiguous()
        gated_shuffled = gated.view(B, C, H, W)

        # Residual connection: project input to expanded channels then add
        skip = self.skip_proj(x)                     # (B, C_exp, H, W)
        out = gated_shuffled + skip                  # (B, C_exp, H, W)

        # Final activation and compact pooling
        out = F.relu(out, inplace=False)
        out = self.final_pool(out)                   # (B, C_exp, final_h, final_w)

        return out

# Default initialization parameters for get_init_inputs()
def get_init_inputs() -> List:
    """
    Returns the initialization parameters to construct Model:
      [in_channels, expand_channels, pool_output_size, final_pool_output, bn_eps, channel_groups]
    """
    return [in_channels, expand_channels, pool_output_size, final_pool_output, bn_eps, channel_groups]

def get_inputs() -> List[torch.Tensor]:
    """
    Creates a random input tensor matching module-level configuration:
      shape = (batch_size, in_channels, height, width)
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]