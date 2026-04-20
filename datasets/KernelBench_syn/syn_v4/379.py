import torch
import torch.nn as nn
from typing import Tuple

class Model(nn.Module):
    """
    Complex 3D -> 2D processing module that:
    - Applies ReflectionPad3d to preserve boundaries
    - A 3D convolution + ReLU extracts volumetric features
    - Applies ConstantPad3d to bias one side of the volume
    - Collapses the depth dimension via mean to produce a 4D tensor (N, C, H, W)
    - Uses FractionalMaxPool2d (with return_indices) to perform fractional spatial pooling
    - Applies a 1x1 Conv2d to mix channels
    - Uses the pooling indices to produce a small, per-channel scaling factor and adjusts the output
    - Returns a compact (N, out_channels) feature vector via global average pooling
    """
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_size3d: int = 3,
        reflect_pad: Tuple[int, int, int, int, int, int] = (1, 1, 1, 1, 1, 1),
        const_pad: Tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 1, 1),
        const_value: float = 0.1,
        frac_kernel_size: Tuple[int, int] = (2, 2),
        frac_output_ratio: Tuple[float, float] = (0.5, 0.5),
    ):
        """
        Initialize the module components.

        Args:
            in_channels (int): Number of input channels for 3D input.
            mid_channels (int): Number of channels after the first Conv3d.
            out_channels (int): Number of output channels for the final representation.
            kernel_size3d (int): Kernel size for Conv3d.
            reflect_pad (tuple): 6-tuple for ReflectionPad3d (L, R, T, B, F, BK) order.
            const_pad (tuple): 6-tuple for ConstantPad3d.
            const_value (float): Constant value for ConstantPad3d.
            frac_kernel_size (tuple): Kernel size for FractionalMaxPool2d.
            frac_output_ratio (tuple): Output ratio for FractionalMaxPool2d (fractional spatial reduction).
        """
        super(Model, self).__init__()
        # Padding layers
        self.reflect_pad = nn.ReflectionPad3d(reflect_pad)
        self.const_pad = nn.ConstantPad3d(const_pad, value=const_value)

        # Volumetric feature extractor
        self.conv3d = nn.Conv3d(in_channels, mid_channels, kernel_size=kernel_size3d, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)

        # Fractional spatial pooling over collapsed depth (2D)
        # return_indices=True to obtain indices and use them for a tiny attention-like scaling
        self.frac_pool2d = nn.FractionalMaxPool2d(kernel_size=frac_kernel_size, output_ratio=frac_output_ratio, return_indices=True)

        # Channel mixing on pooled 2D features
        self.conv2d = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): 5D input tensor of shape (N, C_in, D, H, W).

        Returns:
            torch.Tensor: 2D output tensor of shape (N, out_channels).
        """
        # Reflection padding preserves boundary structure prior to convolution
        x = self.reflect_pad(x)  # -> (N, C_in, D+pad, H+pad, W+pad)

        # 3D conv + activation extracts volumetric features
        x = self.conv3d(x)       # -> (N, mid_channels, D', H', W')
        x = self.relu(x)

        # Constant pad can bias one side (e.g., front/back) to introduce controlled asymmetry
        x = self.const_pad(x)    # -> (N, mid_channels, D'', H'', W'')

        # Collapse depth dimension via mean to obtain a spatial 2D feature map
        # This aggregates volumetric features into a planar representation for fractional pooling
        x2d = x.mean(dim=2)      # -> (N, mid_channels, H'', W'')

        # Fractional Max Pooling returns (output, indices) when return_indices=True
        pooled, indices = self.frac_pool2d(x2d)  # pooled: (N, mid_channels, H_p, W_p); indices same shape (long)

        # 1x1 conv to mix channels post-pooling
        out = self.conv2d(pooled)  # -> (N, out_channels, H_p, W_p)

        # Derive a small per-channel scaling from the pooling indices.
        # indices shape: (N, mid_channels, H_p, W_p). Compute mean over spatial dims,
        # normalize by the original spatial area to keep scaling in a modest range.
        # First, cast to float and compute mean
        idx_mean = indices.float().mean(dim=(2, 3), keepdim=True)  # -> (N, mid_channels, 1, 1)
        # Normalize by a conservative denominator (at least 1)
        denom = float(max(1, x2d.size(2) * x2d.size(3)))
        idx_scale = idx_mean / denom  # -> small values typically in [0, 1]
        # Project idx_scale from mid_channels to out_channels via a simple learned broadcast path:
        # Use average pooling across channel groups by resizing with mean; align channels by simple repeat/interpolate
        # Here we reduce mid_channels -> out_channels scalar per-sample by averaging groups:
        # If channel counts differ, reshape gracefully.
        n, mid_c, _, _ = indices.shape[0], idx_scale.shape[1], 0, 0  # keep typing simple

        # Make idx_scale shape compatible with out channels: average across channel groups then expand
        # Compute channel-wise average to a vector of length out_channels
        # To avoid introducing new parameters, we use simple arithmetic mapping:
        # - If out_channels <= mid_channels: take first out_channels averages
        # - If out_channels > mid_channels: repeat averages to fill
        idx_vec = idx_scale.view(idx_scale.size(0), idx_scale.size(1))  # (N, mid_channels)
        if idx_vec.size(1) >= out.shape[1]:
            idx_vec = idx_vec[:, : out.shape[1]]
        else:
            repeat_factor = (out.shape[1] + idx_vec.size(1) - 1) // idx_vec.size(1)
            idx_vec = idx_vec.repeat(1, repeat_factor)[:, : out.shape[1]]
        # reshape to (N, out_channels, 1, 1) for broadcasting
        idx_vec = idx_vec.view(idx_vec.size(0), idx_vec.size(1), 1, 1).to(out.dtype)

        # Scale the output subtly: out * (1 + small_normalized_index)
        out = out * (1.0 + idx_vec)

        # Final global average pooling to produce a compact vector per sample
        out = out.mean(dim=(2, 3))  # -> (N, out_channels)
        return out


# Module-level configuration variables (example sizes)
batch_size = 8
in_channels = 3
mid_channels = 12
out_channels = 16
depth = 16
height = 64
width = 64

# Padding and pooling configuration that will be passed into the Model constructor via get_init_inputs
kernel_size3d = 3
reflect_pad = (1, 1, 1, 1, 1, 1)            # ReflectionPad3d padding (L, R, T, B, F, BK)
const_pad = (0, 0, 0, 0, 1, 1)              # ConstantPad3d padding
const_value = 0.07                           # Constant value for ConstantPad3d
frac_kernel_size = (2, 2)                    # Kernel size for FractionalMaxPool2d
frac_output_ratio = (0.5, 0.5)               # Fractional reduction ratios

def get_inputs():
    """
    Returns sample inputs for the forward pass.
    The input is a 5D tensor shaped (batch_size, in_channels, depth, height, width).
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Returns the initialization arguments for the Model constructor in order.
    """
    return [
        in_channels,
        mid_channels,
        out_channels,
        kernel_size3d,
        reflect_pad,
        const_pad,
        const_value,
        frac_kernel_size,
        frac_output_ratio,
    ]