import torch
import torch.nn as nn
from typing import Tuple, List

class Model(nn.Module):
    """
    Complex 3D-to-1D processing module that:
    - Applies zero-padding in 3D
    - Collapses spatial dimensions into a 1D temporal-like axis
    - Applies Instance Normalization over the per-channel sequences
    - Performs adaptive max pooling along the sequence axis
    - Reduces pooled features to a per-channel summary and applies a final non-linearity

    This module demonstrates combining nn.ZeroPad3d, nn.InstanceNorm1d,
    and nn.AdaptiveMaxPool1d in a single coherent processing pipeline.
    """
    def __init__(self, pad: Tuple[int, int, int, int, int, int], num_channels: int, pooled_length: int, eps: float = 1e-5, affine: bool = True):
        """
        Args:
            pad (Tuple[int, int, int, int, int, int]): zero pad amounts (left, right, top, bottom, front, back)
            num_channels (int): number of input channels expected
            pooled_length (int): output size for AdaptiveMaxPool1d
            eps (float): epsilon value for InstanceNorm1d
            affine (bool): whether InstanceNorm1d has learnable affine parameters
        """
        super(Model, self).__init__()
        # zero padding in 3D
        self.pad3d = nn.ZeroPad3d(pad)
        # instance normalization treating collapsed spatial dims as sequence length
        self.inst_norm = nn.InstanceNorm1d(num_features=num_channels, eps=eps, affine=affine)
        # adaptive max pooling to a fixed output length
        self.adaptive_pool = nn.AdaptiveMaxPool1d(pooled_length)
        # small learnable scaling parameter per channel for final modulation
        self.channel_scale = nn.Parameter(torch.ones(num_channels), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:

        1. x: (batch, channels, depth, height, width)
        2. apply ZeroPad3d -> (batch, channels, D', H', W')
        3. collapse spatial dims -> (batch, channels, L) where L = D'*H'*W'
        4. instance normalize across each (channel, length)
        5. adaptive max pool to (batch, channels, pooled_length)
        6. compute channel-wise mean across pooled_length -> (batch, channels)
        7. apply learned per-channel scale and tanh nonlinearity -> (batch, channels)

        Returns:
            Tensor of shape (batch, channels) summarizing the input per channel.
        """
        # 1 -> 2: ZeroPad3d
        x_padded = self.pad3d(x)

        # 3: collapse spatial dims into a 1D sequence per channel
        batch = x_padded.shape[0]
        channels = x_padded.shape[1]
        # reshape to (batch, channels, L) where L = product of remaining dims
        x_seq = x_padded.view(batch, channels, -1)

        # 4: InstanceNorm1d across the sequence dimension for each instance and channel
        x_norm = self.inst_norm(x_seq)

        # 5: Adaptive max pooling to fixed length
        x_pooled = self.adaptive_pool(x_norm)  # shape: (batch, channels, pooled_length)

        # 6: Aggregate pooled features to per-channel summaries (mean over pooled length)
        x_summary = torch.mean(x_pooled, dim=2)  # shape: (batch, channels)

        # 7: Apply learned per-channel scaling and non-linearity
        x_scaled = torch.tanh(x_summary * self.channel_scale)

        return x_scaled

# Configuration variables for input generation and initialization
batch_size = 8
channels = 32
depth = 6
height = 14
width = 14
# ZeroPad3d expects 6 values: (left, right, top, bottom, front, back)
pad = (1, 2, 1, 1, 0, 2)
pooled_length = 12

def get_inputs():
    """
    Returns a list with a single input tensor of shape:
    (batch_size, channels, depth, height, width)
    """
    x = torch.randn(batch_size, channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Returns the initialization arguments for Model in order:
    [pad_tuple, num_channels, pooled_length]
    """
    return [pad, channels, pooled_length]