import torch
import torch.nn as nn
from typing import Tuple, List

class Model(nn.Module):
    """
    Complex model that:
    - Applies a 3D adaptive max-pooling to compress depth/height/width.
    - Flattens the pooled depth into channels and uses a ConvTranspose2d (deconvolution)
      to increase spatial resolution.
    - Applies a ReLU activation.
    - Reinterprets deconvolution channels as a new (channels, depth) layout and
      applies a final AdaptiveMaxPool3d to produce the desired output 3D spatial shape.

    Input: Tensor of shape (N, C_in, D_in, H_in, W_in)
    Output: Tensor of shape (N, C_final, D_out, H_out, W_out) as specified by final_pool_output.
    """
    def __init__(
        self,
        in_channels: int,
        pool_output_size: Tuple[int, int, int],
        deconv_out_channels: int,
        final_depth: int,
        final_pool_output: Tuple[int, int, int],
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1
    ):
        """
        Args:
            in_channels (int): Number of channels in the input.
            pool_output_size (tuple): (D_pool, H_pool, W_pool) target for first AdaptiveMaxPool3d.
            deconv_out_channels (int): Number of output channels for ConvTranspose2d.
            final_depth (int): Desired depth after reshaping deconv output (must divide deconv_out_channels).
            final_pool_output (tuple): (D_out, H_out, W_out) target for final AdaptiveMaxPool3d.
            kernel_size/stride/padding: Parameters for ConvTranspose2d.
        """
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.pool_output_size = pool_output_size
        self.final_depth = final_depth
        self.final_pool_output = final_pool_output

        # Ensure reshape is possible
        if deconv_out_channels % final_depth != 0:
            raise ValueError("deconv_out_channels must be divisible by final_depth to reshape properly")

        # First stage: 3D adaptive max pooling
        self.pool3d = nn.AdaptiveMaxPool3d(output_size=pool_output_size)

        # ConvTranspose2d will operate on a 4D tensor where channels = in_channels * pooled_depth
        pooled_depth = pool_output_size[0]
        conv_in_channels = in_channels * pooled_depth

        # Deconvolution to expand spatial resolution (H,W)
        self.deconv2d = nn.ConvTranspose2d(
            in_channels=conv_in_channels,
            out_channels=deconv_out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        # Non-linearity
        self.relu = nn.ReLU(inplace=True)

        # Final 3D adaptive pooling to produce requested (D_out, H_out, W_out)
        self.final_pool3d = nn.AdaptiveMaxPool3d(output_size=final_pool_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        1. x: (N, C_in, D_in, H_in, W_in)
        2. p = pool3d(x) -> (N, C_in, D_p, H_p, W_p)
        3. reshape to 4D: r = p.view(N, C_in * D_p, H_p, W_p)
        4. y = deconv2d(r) -> (N, deconv_out_channels, H_out, W_out)
        5. y = relu(y)
        6. reshape to 5D by splitting channels into (C_mid, D_mid, H_out, W_out) where D_mid = final_depth
           C_mid = deconv_out_channels // final_depth
        7. apply final_pool3d -> final (N, C_mid, D_out, H_final_out, W_final_out)
        """
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input (N,C,D,H,W), got shape {tuple(x.shape)}")

        N = x.shape[0]
        # 1. initial 3D pooling
        p = self.pool3d(x)  # (N, C_in, D_p, H_p, W_p)

        # 2. collapse depth into channels to run ConvTranspose2d
        N, C_in, D_p, H_p, W_p = p.shape
        r = p.view(N, C_in * D_p, H_p, W_p)  # (N, C_in * D_p, H_p, W_p)

        # 3. deconvolution over 2D spatial dims
        y = self.deconv2d(r)  # (N, deconv_out_channels, H_out, W_out)
        y = self.relu(y)

        # 4. reinterpret channels -> split into (C_mid, final_depth)
        deconv_out_channels = y.shape[1]
        if deconv_out_channels % self.final_depth != 0:
            # This should not happen because of constructor check, but guard anyway
            raise RuntimeError("deconv_out_channels not divisible by final_depth at runtime")

        C_mid = deconv_out_channels // self.final_depth
        H_out = y.shape[2]
        W_out = y.shape[3]

        y5d = y.view(N, C_mid, self.final_depth, H_out, W_out)  # (N, C_mid, D_mid, H_out, W_out)

        # 5. final adaptive max pool to shape (D_out, H_out_final, W_out_final)
        out = self.final_pool3d(y5d)  # (N, C_mid, D_out, H_final, W_final)
        return out

# Module-level configuration variables
batch_size = 8
in_channels = 3
depth = 16
height = 32
width = 32

# First pooling target: reduce depth and spatial dims moderately
pool_output_size = (4, 8, 8)  # (D_p, H_p, W_p)

# Parameters for ConvTranspose2d
deconv_out_channels = 32  # must be divisible by final_depth
kernel_size = 4
stride = 2
padding = 1

# How many depth slices to create after deconv channels are split
final_depth = 4  # deconv_out_channels % final_depth == 0 -> 32 % 4 == 0

# Final desired 3D output spatial resolution
final_pool_output = (2, 8, 8)  # (D_out, H_out, W_out)

def get_inputs():
    """
    Returns:
        List[torch.Tensor]: single-element list containing the input 5D tensor.
                           Shape: (batch_size, in_channels, depth, height, width)
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters for the Model constructor in the same order.
    """
    return [
        in_channels,
        pool_output_size,
        deconv_out_channels,
        final_depth,
        final_pool_output,
        kernel_size,
        stride,
        padding
    ]