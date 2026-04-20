import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Any, Tuple

"""
Complex 3D convolutional module combining ConstantPad3d, Conv3d and Softmax.

Computation pattern:
- ConstantPad3d to ensure controlled boundary behavior.
- Initial Conv3d (with external padding) to produce mid-level feature maps.
- Depthwise Conv3d for channel-wise spatial processing.
- Softmax applied over flattened spatial dimensions per channel to generate
  spatial attention-like weights.
- Reweighting of the features by the softmax maps, residual addition, and
  final 1x1x1 Conv3d projection to produce the output channels.
"""

# Configuration / default parameters at module level
batch_size = 4
in_channels = 3
mid_channels = 16
out_channels = 8
depth = 10
height = 16
width = 16

# Convolution parameters
kernel_size = (3, 3, 3)  # kernel for the first conv (will use external padding)
stride = (1, 2, 2)       # stride for the first conv (reduces spatial H/W)
external_pad = (1, 1, 1, 1, 1, 1)  # ConstantPad3d pad: (L, R, T, B, F, Ba)
pad_value = 0.0          # constant value for padding


class Model(nn.Module):
    """
    A PyTorch model that demonstrates a multi-stage 3D convolutional block.
    - Pads the input with ConstantPad3d.
    - Applies a Conv3d (treated as the main feature extractor).
    - Applies a depthwise Conv3d to process each channel spatially.
    - Computes spatial softmax per channel, reweights features, adds a residual
      connection, and projects to the desired output channels with a 1x1x1 conv.
    """

    def __init__(
        self,
        in_ch: int,
        mid_ch: int,
        out_ch: int,
        kernel: Tuple[int, int, int] = (3, 3, 3),
        stride: Tuple[int, int, int] = (1, 1, 1),
        pad: Tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0),
        pad_val: float = 0.0,
    ):
        """
        Initializes the module.

        Args:
            in_ch (int): Number of input channels.
            mid_ch (int): Number of intermediate channels (for internal convs).
            out_ch (int): Number of output channels.
            kernel (tuple): Kernel size for the initial Conv3d.
            stride (tuple): Stride for the initial Conv3d.
            pad (tuple): 6-element tuple for ConstantPad3d: (L, R, T, B, F, Ba).
            pad_val (float): Constant value used by ConstantPad3d.
        """
        super(Model, self).__init__()

        # External explicit padding layer
        self.pad = nn.ConstantPad3d(pad, pad_val)

        # Primary conv: we set padding=0 because we handle it with ConstantPad3d
        self.conv1 = nn.Conv3d(in_ch, mid_ch, kernel_size=kernel, stride=stride, padding=0, bias=False)

        # Depthwise conv: groups=mid_ch (depthwise) to process spatial info per channel
        # Use padding=1 to maintain spatial dims after depthwise conv (kernel_size=3)
        self.conv_dw = nn.Conv3d(mid_ch, mid_ch, kernel_size=3, stride=1, padding=1, groups=mid_ch, bias=False)

        # Final projection conv is 1x1x1 to mix channel information after reweighting
        self.conv_proj = nn.Conv3d(mid_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True)

        # Softmax applied over the flattened spatial dimension per channel
        # We'll instantiate it with dim=-1 and apply to appropriately shaped tensors
        self.softmax = nn.Softmax(dim=-1)

        # small learnable scaling of the attention branch (helps training stability)
        self.attn_scale = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the complex 3D block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, D, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (B, out_ch, D', H', W')
        """
        # Step 1: explicit constant padding
        x_padded = self.pad(x)  # shape changed according to external_pad

        # Step 2: primary convolutional feature extraction
        feat = self.conv1(x_padded)  # (B, mid_ch, D1, H1, W1)
        feat = F.relu(feat)

        # Step 3: depthwise spatial processing
        feat_dw = self.conv_dw(feat)  # (B, mid_ch, D1, H1, W1)
        feat_dw = F.relu(feat_dw)

        # Step 4: spatial softmax per-channel
        b, c, d, h, w = feat_dw.shape
        spatial_flat = feat_dw.view(b, c, -1)  # (B, mid_ch, N) where N = D1*H1*W1

        # Compute per-channel spatial attention maps (softmax over N)
        attn = self.softmax(spatial_flat)  # (B, mid_ch, N)

        # Optionally scale the attention and reweight features
        attn = self.attn_scale * attn

        reweighted_flat = spatial_flat * attn  # elementwise reweighting (B, mid_ch, N)
        reweighted = reweighted_flat.view(b, c, d, h, w)  # (B, mid_ch, D1, H1, W1)

        # Step 5: residual connection and projection
        aggregated = reweighted + feat_dw  # residual fusion
        out = self.conv_proj(aggregated)  # project to out_ch

        return out


def get_inputs() -> List[torch.Tensor]:
    """
    Returns a sample input tensor list appropriate for this model.

    Output:
        [x] where x has shape (batch_size, in_channels, depth, height, width)
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]


def get_init_inputs() -> List[Any]:
    """
    Returns the initialization arguments for the Model class.

    Returns:
        [in_channels, mid_channels, out_channels, kernel_size, stride, external_pad, pad_value]
    """
    return [in_channels, mid_channels, out_channels, kernel_size, stride, external_pad, pad_value]