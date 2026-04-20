import torch
import torch.nn as nn
from typing import Tuple, List

class Model(nn.Module):
    """
    Volumetric-to-image processing module that:
    - Pads a 3D volume (ZeroPad3d)
    - Collapses the depth dimension via mean reduction
    - Upsamples the resulting 2D feature maps using bilinear interpolation
    - Applies a SELU non-linearity followed by AlphaDropout

    This combines spatial padding across 3 dims with a depth reduction,
    then performs 2D operations on the collapsed feature maps.
    """
    def __init__(self, pad: Tuple[int, int, int, int, int, int], scale_factor: int = 2, dropout_p: float = 0.1):
        """
        Args:
            pad (tuple): ZeroPad3d padding in the order
                         (padW_left, padW_right, padH_top, padH_bottom, padD_front, padD_back)
            scale_factor (int): Integer upsampling factor for H and W after depth reduction.
            dropout_p (float): Dropout probability for AlphaDropout.
        """
        super(Model, self).__init__()
        # ZeroPad3d pads (W_left, W_right, H_top, H_bottom, D_front, D_back)
        self.pad3d = nn.ZeroPad3d(pad)
        # Upsampling for 2D feature maps after depth reduction
        self.upsample2d = nn.UpsamplingBilinear2d(scale_factor=scale_factor)
        # AlphaDropout for preserving SELU self-normalizing properties
        self.alpha_dropout = nn.AlphaDropout(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation:
          1. Pad the volumetric tensor (N, C, D, H, W) -> (N, C, D', H', W')
          2. Reduce depth via mean: -> (N, C, H', W')
          3. Upsample spatial dims with bilinear interpolation -> (N, C, H_up, W_up)
          4. Apply SELU activation
          5. Apply AlphaDropout

        Args:
            x (torch.Tensor): Input volumetric tensor of shape (N, C, D, H, W)

        Returns:
            torch.Tensor: Processed tensor of shape (N, C, H_up, W_up)
        """
        # 1) Pad volume
        x_padded = self.pad3d(x)

        # 2) Collapse depth dimension by taking mean across D axis
        # x_padded is (N, C, D_p, H_p, W_p) -> mean over dim=2 -> (N, C, H_p, W_p)
        x_collapsed = torch.mean(x_padded, dim=2)

        # 3) Upsample spatial resolution using bilinear interpolation
        x_up = self.upsample2d(x_collapsed)

        # 4) Non-linearity (SELU) to encourage self-normalizing activations
        x_activated = torch.selu(x_up)

        # 5) AlphaDropout (keeps mean/std properties for SELU)
        x_dropped = self.alpha_dropout(x_activated)

        return x_dropped

# Configuration for synthetic inputs
batch_size = 8
channels = 3
depth = 5
height = 64
width = 48

# Initialization parameters for the Model
# Padding: (padW_left, padW_right, padH_top, padH_bottom, padD_front, padD_back)
pad = (1, 2, 3, 1, 0, 1)
scale_factor = 2
dropout_p = 0.12

def get_inputs() -> List[torch.Tensor]:
    """
    Create a random volumetric input tensor with shape (N, C, D, H, W).

    Returns:
        list: Single-element list containing the input tensor.
    """
    x = torch.randn(batch_size, channels, depth, height, width)
    return [x]

def get_init_inputs() -> List:
    """
    Provides initialization parameters used to construct the Model.

    Returns:
        list: [pad, scale_factor, dropout_p]
    """
    return [pad, scale_factor, dropout_p]