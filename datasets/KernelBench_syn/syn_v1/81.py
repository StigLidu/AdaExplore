import torch
import torch.nn as nn
from typing import Tuple, List

class Model(nn.Module):
    """
    Complex 3D-to-2D reduction and feature projection model.

    Pipeline:
    1. ReflectionPad3d on the 5D input (N, C, D, H, W).
    2. Collapse the depth dimension via mean to produce a 4D tensor (N, C, H, W).
    3. Apply a LazyConv2d to produce spatial feature maps (Lazy in_channels inference).
    4. Non-linear activation (GELU).
    5. Rearrange and apply a LazyLinear per-spatial-location (inferred in_features).
    6. Per-sample normalization over spatial+channel dims and return as (N, out_features, H, W).
    """
    def __init__(
        self,
        conv_out_channels: int,
        linear_out_features: int,
        conv_kernel: int = 3,
        pad: Tuple[int, int, int, int, int, int] = (1, 1, 1, 1, 0, 0)
    ):
        """
        Args:
            conv_out_channels (int): Number of output channels for the 2D convolution.
            linear_out_features (int): Output features of the linear projection applied per spatial location.
            conv_kernel (int): Kernel size for the 2D convolution.
            pad (tuple): 6-tuple padding for ReflectionPad3d (left, right, top, bottom, front, back).
        """
        super(Model, self).__init__()
        # 3D reflection padding (will operate on (N, C, D, H, W))
        self.pad3d = nn.ReflectionPad3d(pad)
        # LazyConv2d will infer in_channels on first forward pass
        self.conv2d = nn.LazyConv2d(out_channels=conv_out_channels, kernel_size=conv_kernel, padding=conv_kernel // 2)
        self.act = nn.GELU()
        # LazyLinear will infer in_features on first forward pass when given (N*H*W, in_features)
        self.proj = nn.LazyLinear(out_features=linear_out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor shape (N, C, D, H, W).

        Returns:
            torch.Tensor: Output tensor shape (N, linear_out_features, H', W') where H' and W' depend on padding.
        """
        # x: (N, C, D, H, W)
        # 1) Reflection pad in 3D
        x_p = self.pad3d(x)  # still (N, C, D_p, H_p, W_p)

        # 2) Collapse the depth dimension by averaging
        # Choose depth dimension index 2
        x2 = torch.mean(x_p, dim=2)  # -> (N, C, H_p, W_p)

        # 3) 2D convolution (LazyConv2d infers in_channels)
        x3 = self.conv2d(x2)  # -> (N, C_conv, H_p, W_p)

        # 4) Non-linearity
        x4 = self.act(x3)  # -> (N, C_conv, H_p, W_p)

        # 5) Rearrange to apply linear per spatial position
        N, C_conv, H_p, W_p = x4.shape
        # Move channels to last dim and flatten spatial dims
        x_perm = x4.permute(0, 2, 3, 1).contiguous()  # (N, H_p, W_p, C_conv)
        x_flat = x_perm.view(N * H_p * W_p, C_conv)  # (N*H_p*W_p, C_conv)

        # 6) Linear projection (LazyLinear infers in_features)
        x_lin = self.proj(x_flat)  # (N*H_p*W_p, linear_out_features)
        out_features = x_lin.shape[-1]
        x_unflat = x_lin.view(N, H_p, W_p, out_features)  # (N, H_p, W_p, out_features)

        # 7) Per-sample normalization across spatial and feature dims
        # Compute L2 norm across (H_p, W_p, out_features) for each sample
        norms = torch.norm(x_unflat, p=2, dim=(1, 2, 3), keepdim=True)  # (N, 1, 1, 1)
        x_norm = x_unflat / (norms + 1e-6)

        # 8) Permute to channel-first (N, out_features, H_p, W_p) for standard conv-like output
        out = x_norm.permute(0, 3, 1, 2).contiguous()
        return out

# Configuration / default parameters
batch_size = 8
in_channels = 3
depth = 4
height = 32
width = 32

conv_out_channels = 16
linear_out_features = 64
conv_kernel = 3
pad3d = (1, 1, 1, 1, 0, 0)  # (left, right, top, bottom, front, back)

def get_inputs() -> List[torch.Tensor]:
    """
    Returns a list containing a single input tensor shaped (batch_size, in_channels, depth, height, width).
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs() -> List:
    """
    Returns the initialization parameters for the Model constructor:
    [conv_out_channels, linear_out_features, conv_kernel, pad3d]
    """
    return [conv_out_channels, linear_out_features, conv_kernel, pad3d]