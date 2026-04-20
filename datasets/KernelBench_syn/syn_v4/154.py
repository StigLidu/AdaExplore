import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# Configuration variables
BATCH_SIZE = 8
CHANNELS = 32
HEIGHT = 64
WIDTH = 64
PAD_SIZE = 2
SCALE_FACTOR = 2
INSTANCENORM_AFFINE = True

class Model(nn.Module):
    """
    Complex image-like processing module that combines:
      - Reflection padding
      - Instance normalization
      - Channel-wise gating computed from global pooled descriptors via a learned channel mixing matrix
      - Upsampling (bilinear)
      - Spatial flip-based residual mixing

    The module demonstrates a chained set of tensor transformations mixing spatial and channel operations.
    """
    def __init__(self, channels: int = CHANNELS, pad: int = PAD_SIZE, scale: int = SCALE_FACTOR, affine: bool = INSTANCENORM_AFFINE):
        super(Model, self).__init__()
        self.channels = channels
        self.pad = pad
        self.scale = scale

        # Layers
        self.reflection_pad = nn.ReflectionPad2d(self.pad)
        self.inst_norm = nn.InstanceNorm2d(self.channels, affine=affine)
        # Upsample spatially (bilinear for 2D data)
        self.upsample = nn.Upsample(scale_factor=self.scale, mode='bilinear', align_corners=False)

        # Learned channel mixing matrix used to compute gating from global pooled features
        # Shape: (C, C), used as gap (B, C) @ channel_weight (C, C) -> (B, C)
        self.channel_weight = nn.Parameter(torch.randn(self.channels, self.channels) * 0.02)

        # Small initialization tweak for stability
        nn.init.eye_(self.channel_weight)  # start as near-identity to preserve initial information

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the composed operations.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Processed tensor of shape (B, C, H*scale, W*scale)
        """
        # 1) Reflection padding increases receptive field without introducing artificial zeros
        x_pad = self.reflection_pad(x)  # (B, C, H + 2*pad, W + 2*pad)

        # 2) Instance normalization normalizes per-instance, per-channel
        x_norm = self.inst_norm(x_pad)  # (B, C, H', W')

        # 3) Global average pooling across spatial dims -> compact descriptor per channel (B, C)
        gap = x_norm.mean(dim=(2, 3))  # (B, C)

        # 4) Channel mixing: learn interactions between channels to produce gating coefficients
        #    Use learned channel_weight to mix channels; apply non-linearity
        mixed = torch.matmul(gap, self.channel_weight)  # (B, C)
        gating = torch.sigmoid(mixed).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)

        # 5) Channel-wise gating applied to normalized features
        x_gated = x_norm * gating  # (B, C, H', W')

        # 6) Upsample to increase spatial resolution
        x_up = self.upsample(x_gated)  # (B, C, H'*scale, W'*scale)

        # 7) Spatial flip-based residual mixing: reflect spatial content and blend
        x_flip = torch.flip(x_up, dims=[2, 3])  # flip height and width
        out = 0.6 * x_up + 0.4 * x_flip  # (B, C, H_out, W_out)

        # 8) Final lightweight non-linearity to bound outputs
        return torch.tanh(out)


def get_inputs() -> List[torch.Tensor]:
    """
    Generates a batch of synthetic image-like tensors for the model.

    Returns:
        list: [x] where x is a tensor of shape (BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)
    """
    x = torch.randn(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)
    return [x]


def get_init_inputs() -> List:
    """
    Returns inputs needed to instantiate the Model (arguments to __init__).
    This makes it explicit what configuration the module will be created with.

    Returns:
        list: [channels, pad, scale, affine]
    """
    return [CHANNELS, PAD_SIZE, SCALE_FACTOR, INSTANCENORM_AFFINE]