import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

class Model(nn.Module):
    """
    Complex 3D-aware module that:
    - Accepts a 5D tensor (N, C, D, H, W)
    - Applies PixelUnshuffle by merging the depth into the batch dimension,
      performing 2D PixelUnshuffle, and restoring the 5D shape with expanded channels
    - Normalizes across 3D channels with BatchNorm3d
    - Applies an AdaptiveMaxPool3d to a target (d_out, h_out, w_out)
    - Builds a channel-wise gating signal from per-channel maxima and uses it to
      modulate pooled features
    - Produces a final dense representation through a fully-connected layer
    """
    def __init__(
        self,
        in_channels: int,
        unshuffle_factor: int = 2,
        pool_output: Tuple[int, int, int] = (4, 8, 8),
        fc_output_dim: int = 1024
    ):
        """
        Args:
            in_channels: number of input channels C
            unshuffle_factor: downscale factor for PixelUnshuffle (r)
            pool_output: desired (D_out, H_out, W_out) for AdaptiveMaxPool3d
            fc_output_dim: final output vector dimension
        """
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.r = unshuffle_factor
        self.pool_output = pool_output  # (d_out, h_out, w_out)
        self.fc_output_dim = fc_output_dim

        # PixelUnshuffle operates on 4D (N, C, H, W), so we will merge depth into batch,
        # apply PixelUnshuffle, and then restore a 5D tensor with expanded channels.
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=self.r)

        # After PixelUnshuffle the channels become in_channels * r^2
        self.expanded_channels = in_channels * (self.r ** 2)

        # BatchNorm3d operates on (N, C, D, H, W) style inputs
        self.bn3d = nn.BatchNorm3d(num_features=self.expanded_channels)

        # Adaptive max pooling over 3D spatial dims to the configured output size
        self.adaptive_pool = nn.AdaptiveMaxPool3d(output_size=self.pool_output)

        # Channel gating: project per-channel maxima to produce a gating vector
        self.gate_fc = nn.Linear(self.expanded_channels, self.expanded_channels)

        # Final fully connected projection from flattened pooled features
        pooled_flat_dim = self.expanded_channels * self.pool_output[0] * self.pool_output[1] * self.pool_output[2]
        self.fc = nn.Linear(pooled_flat_dim, self.fc_output_dim)

        # Small initialization to stabilize gating and final projection
        nn.init.kaiming_uniform_(self.gate_fc.weight, a=0.1)
        nn.init.zeros_(self.gate_fc.bias)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (N, C, D, H, W)

        Returns:
            Tensor of shape (N, fc_output_dim)
        """
        # Validate input dims
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input (N,C,D,H,W), got {x.dim()}D tensor")

        N, C, D, H, W = x.shape
        if C != self.in_channels:
            raise ValueError(f"Input channel mismatch: expected {self.in_channels}, got {C}")
        if H % self.r != 0 or W % self.r != 0:
            raise ValueError(f"H and W must be divisible by unshuffle factor {self.r}")

        # Merge depth into batch to use 2D PixelUnshuffle
        # (N, C, D, H, W) -> (N, D, C, H, W) -> (N*D, C, H, W)
        x_merged = x.permute(0, 2, 1, 3, 4).contiguous().view(N * D, C, H, W)

        # Apply PixelUnshuffle: (N*D, C, H, W) -> (N*D, C*r^2, H/r, W/r)
        x_unshuffled = self.pixel_unshuffle(x_merged)

        # Restore 5D shape: (N*D, C*r^2, H/r, W/r) -> (N, D, C*r^2, H/r, W/r)
        H_r = H // self.r
        W_r = W // self.r
        x5d = x_unshuffled.view(N, D, self.expanded_channels, H_r, W_r)

        # Permute to (N, C_expanded, D, H_r, W_r) for BatchNorm3d
        x5d = x5d.permute(0, 2, 1, 3, 4).contiguous()

        # BatchNorm3d
        x_norm = self.bn3d(x5d)

        # Non-linearity
        x_act = F.relu(x_norm, inplace=True)

        # Adaptive 3D max pooling to reduce to configured spatial dims
        x_pooled = self.adaptive_pool(x_act)  # (N, C_expanded, D_out, H_out, W_out)

        # Channel-wise global maxima across spatial sites to build gating
        # (N, C_expanded, D_out*H_out*W_out) -> max over spatial dims -> (N, C_expanded)
        spatial_flat = x_pooled.view(N, self.expanded_channels, -1)
        channel_max = spatial_flat.max(dim=2).values  # (N, C_expanded)

        # Gate: project and sigmoid to obtain per-channel gates in (0,1)
        gate = torch.sigmoid(self.gate_fc(channel_max))  # (N, C_expanded)

        # Apply gating by broadcasting over spatial dims
        gate_reshaped = gate.view(N, self.expanded_channels, 1, 1, 1)
        x_modulated = x_pooled * gate_reshaped

        # Flatten pooled features and apply final fully-connected layer
        x_final = x_modulated.view(N, -1)  # (N, expanded_channels * D_out * H_out * W_out)
        out = self.fc(x_final)  # (N, fc_output_dim)

        return out

# Configuration variables
batch_size = 8
in_channels = 16
depth = 8
height = 32
width = 32
unshuffle_factor = 2
pool_output = (4, 8, 8)
fc_output_dim = 1024

def get_inputs():
    """
    Returns a list with a single 5D input tensor suitable for Model.forward:
    shape = (batch_size, in_channels, depth, height, width)
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for Model:
    [in_channels, unshuffle_factor, pool_output (tuple), fc_output_dim]
    """
    return [in_channels, unshuffle_factor, pool_output, fc_output_dim]