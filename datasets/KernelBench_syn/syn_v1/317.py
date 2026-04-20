import torch
import torch.nn as nn
from typing import Tuple, List

# Configuration variables
batch_size = 8
channels = 16
depth = 8
height = 64
width = 64

# Unfold params (kernel over HxW), stride over HxW, symmetric pad over D,H,W
kernel_size = (3, 3)
stride = (2, 2)
pad_sizes = (1, 2, 2)  # (pad_d, pad_h, pad_w)


class Model(nn.Module):
    """
    Complex model that:
    - Applies 3D zero padding to a volumetric input (N, C, D, H, W)
    - Merges the depth into the channel dimension to create a 4D tensor suitable for nn.Unfold
    - Extracts sliding local blocks with nn.Unfold
    - Computes spatial attention weights with nn.Softmax across the patch locations
    - Aggregates patches into a per-channel descriptor, combines it with global average pooling,
      and uses a learned linear gating to produce the final per-channel output vector.
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        pad_sizes: Tuple[int, int, int],
    ):
        """
        Args:
            in_channels (int): Number of input channels C.
            kernel_size (Tuple[int,int]): Kernel size for unfolding over HxW.
            stride (Tuple[int,int]): Stride for unfolding over HxW.
            pad_sizes (Tuple[int,int,int]): Symmetric padding for D,H,W respectively.
        """
        super(Model, self).__init__()

        self.in_channels = in_channels
        self.kH, self.kW = kernel_size
        self.sH, self.sW = stride
        self.pad_d, self.pad_h, self.pad_w = pad_sizes

        # ZeroPad3d expects (padWLeft, padWRight, padHLeft, padHRight, padDLeft, padDRight)
        pad_3d = (
            self.pad_w,
            self.pad_w,
            self.pad_h,
            self.pad_h,
            self.pad_d,
            self.pad_d,
        )
        self.pad3d = nn.ZeroPad3d(pad_3d)

        # Unfold over HxW on the reshaped (N, C*D, H, W) tensor
        self.unfold = nn.Unfold(kernel_size=(self.kH, self.kW), stride=(self.sH, self.sW))

        # Softmax to produce attention weights across spatial locations (L)
        self.softmax_spatial = nn.Softmax(dim=2)

        # A small projection to compute gating between aggregated patch summary and global avg
        # Input dim = 2 * C (aggregated + global), output dim = C (per-channel gate pre-activation)
        self.channel_proj = nn.Linear(2 * in_channels, in_channels)

        # Non-linearity for gating
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (N, C, D, H, W)

        Returns:
            Tensor of shape (N, C) representing per-channel descriptors after attention and gating.
        """
        # Step 1: Zero-pad the volumetric input
        # x_pad: (N, C, D2, H2, W2)
        x_pad = self.pad3d(x)

        N, C, D2, H2, W2 = x_pad.shape

        # Step 2: Merge depth dimension into channels to obtain a 4D tensor for unfolding
        # reshaped: (N, C * D2, H2, W2)
        merged = x_pad.reshape(N, C * D2, H2, W2)

        # Step 3: Extract sliding local blocks -> patches: (N, (C*D2)*kH*kW, L)
        patches = self.unfold(merged)

        # Step 4: Spatial attention weights across sliding locations L
        # attn: (N, (C*D2)*kH*kW, L)
        attn = self.softmax_spatial(patches)

        # Step 5: Weighted aggregation across spatial locations -> (N, (C*D2)*kH*kW)
        aggregated_flat = (patches * attn).sum(dim=2)

        # Step 6: Convert aggregated_flat to per-channel summary
        # aggregated_flat reshaped to (N, C, D2 * kH * kW)
        agg_per_channel = aggregated_flat.view(N, C, D2 * self.kH * self.kW)

        # Mean over the combined depth & patch-element dimension -> (N, C)
        agg_summary = agg_per_channel.mean(dim=2)

        # Path B: Global average pooling over the original volumetric dimensions -> (N, C)
        global_avg = x.mean(dim=(2, 3, 4))

        # Step 7: Learned gating to combine the two summaries
        # Concatenate (N, 2*C), linear -> (N, C), sigmoid -> gate (N, C)
        combined = torch.cat([agg_summary, global_avg], dim=1)
        gate_pre = self.channel_proj(combined)
        gate = self.sigmoid(gate_pre)

        # Final mix: gated combination favoring aggregated summary where gate is high
        out = agg_summary * gate + global_avg * (1.0 - gate)

        return out


def get_inputs() -> List[torch.Tensor]:
    """
    Returns a list with the primary input tensor for the model:
    - x: random volumetric tensor of shape (batch_size, channels, depth, height, width)
    """
    x = torch.randn(batch_size, channels, depth, height, width)
    return [x]


def get_init_inputs() -> List:
    """
    Returns the initialization parameters for the Model constructor:
    [in_channels, kernel_size, stride, pad_sizes]
    """
    return [channels, kernel_size, stride, pad_sizes]