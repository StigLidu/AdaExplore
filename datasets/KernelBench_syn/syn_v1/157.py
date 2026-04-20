import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

"""
Complex example combining LPPool2d, Softmax2d and Fold/Unfold patch processing.

Computation pipeline:
1. Project input channels -> mid_channels with 1x1 conv.
2. Apply LPPool2d to obtain a compact spatial attention map.
3. Softmax2d over channels at each spatial location to form channel-wise attention.
4. Upsample attention to the original spatial resolution and modulate projected feature map.
5. Extract overlapping patches with Unfold, perform a simple per-patch modulation,
   and reconstruct spatial map with Fold.
6. Reproject to original in_channels and add a residual connection to the input.
"""

# Configuration variables
batch_size = 8
in_channels = 3
mid_channels = 16
height = 64
width = 64

# LPPool settings
lp_norm = 2  # p-norm for LPPool2d
pool_kernel = (2, 2)
pool_stride = (2, 2)

# Patch (Unfold/Fold) settings
fold_kernel = (3, 3)
fold_stride = (2, 2)
fold_padding = 1  # padding used for both unfold and fold to maintain shapes


class Model(nn.Module):
    """
    Model combining LPPool2d, Softmax2d and Fold-based patch reconstruction.

    Args:
        in_ch (int): Number of input channels.
        mid_ch (int): Number of intermediate channels after projection.
        lp (int): Norm type for LPPool2d.
        pool_k (Tuple[int, int]): Kernel size for LPPool2d.
        pool_s (Tuple[int, int]): Stride for LPPool2d.
        patch_k (Tuple[int, int]): Kernel size for Unfold/Fold patches.
        patch_s (Tuple[int, int]): Stride for Unfold/Fold patches.
        pad (int): Padding for Unfold/Fold.
        out_h (int): Target output height for fold/unfold.
        out_w (int): Target output width for fold/unfold.
    """
    def __init__(
        self,
        in_ch: int,
        mid_ch: int,
        lp: int,
        pool_k: Tuple[int, int],
        pool_s: Tuple[int, int],
        patch_k: Tuple[int, int],
        patch_s: Tuple[int, int],
        pad: int,
        out_h: int,
        out_w: int
    ):
        super(Model, self).__init__()
        # Project input into a richer feature space
        self.proj = nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=True)

        # Lp pooling to obtain compact spatial summaries
        # nn.LPPool2d(norm_type, kernel_size, stride=None, ceil_mode=False)
        self.lppool = nn.LPPool2d(lp, pool_k, stride=pool_s)

        # Softmax per spatial location (over channels)
        self.softmax2d = nn.Softmax2d()

        # Unfold / Fold for patchwise processing
        self.unfold = nn.Unfold(kernel_size=patch_k, stride=patch_s, padding=pad)
        self.fold = nn.Fold(output_size=(out_h, out_w), kernel_size=patch_k, stride=patch_s, padding=pad)

        # Reproject features back to original input channels for residual connection
        self.reproj = nn.Conv2d(mid_ch, in_ch, kernel_size=1, bias=True)

        # Store shapes for later use
        self.mid_ch = mid_ch
        self.patch_k = patch_k
        self.patch_s = patch_s
        self.out_h = out_h
        self.out_w = out_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (N, in_channels, H, W)

        Returns:
            torch.Tensor: Output tensor of same shape as input (residual + processed).
        """
        # Project input
        proj = self.proj(x)  # (N, mid_ch, H, W)

        # Lp pooling -> reduced spatial resolution: (H', W')
        pooled = self.lppool(proj)  # (N, mid_ch, H_p, W_p)

        # Spatial softmax over channels at each (h, w): resulting attention sums to 1 across channels
        att = self.softmax2d(pooled)  # (N, mid_ch, H_p, W_p)

        # Upsample attention to original projected spatial resolution
        att_up = F.interpolate(att, size=(proj.shape[2], proj.shape[3]), mode='bilinear', align_corners=False)
        # Modulate projected features by attention (channel-wise, per spatial location)
        attended = proj * att_up  # (N, mid_ch, H, W)

        # Extract overlapping patches
        patches = self.unfold(attended)  # (N, mid_ch * kH * kW, L), where L is number of sliding blocks

        # Compute a simple per-patch statistic and use it to non-linearly modulate patches
        # patch_means shape: (N, 1, L)
        patch_means = patches.mean(dim=1, keepdim=True)
        # Use a sigmoid gating based on mean to scale patches
        gates = torch.sigmoid(patch_means)
        patches_mod = patches * gates  # (N, mid_ch * kH * kW, L)

        # Reconstruct spatial map from modified patches
        reconstructed = self.fold(patches_mod)  # (N, mid_ch, H, W)

        # Reproject back to input channels and add residual connection
        out = self.reproj(reconstructed) + x  # (N, in_ch, H, W)
        return out


def get_inputs() -> List[torch.Tensor]:
    """
    Returns a list containing one sample input tensor suitable for the model.

    Shape: (batch_size, in_channels, height, width)
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs() -> List:
    """
    Returns initialization parameters for the Model constructor in the same order.

    Order:
        in_ch, mid_ch, lp, pool_k, pool_s, patch_k, patch_s, pad, out_h, out_w
    """
    return [
        in_channels,
        mid_channels,
        lp_norm,
        pool_kernel,
        pool_stride,
        fold_kernel,
        fold_stride,
        fold_padding,
        height,
        width,
    ]