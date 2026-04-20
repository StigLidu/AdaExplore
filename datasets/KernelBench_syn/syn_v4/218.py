import torch
import torch.nn as nn
from typing import List, Tuple, Any

class Model(nn.Module):
    """
    Complex 3D-to-vector module that:
    - Applies ReflectionPad3d to spatial dimensions
    - Flattens spatial dimensions into a single sequence
    - Applies AdaptiveMaxPool1d over that sequence to reduce length
    - Applies a Softplus activation
    - Projects the resulting per-channel pooled features into an output vector via Linear

    Input: Tensor of shape (N, C, D, H, W)
    Output: Tensor of shape (N, out_features)
    """
    def __init__(
        self,
        pool_output_size: int = 32,
        pad_sizes: Tuple[int, int, int] = (1, 2, 1),
        in_channels: int = 3,
        out_features: int = 128,
        softplus_beta: float = 1.0,
    ):
        """
        Initializes the module.

        Args:
            pool_output_size: Number of output locations for AdaptiveMaxPool1d along the flattened spatial axis.
            pad_sizes: Tuple of symmetric paddings (pad_depth, pad_height, pad_width). Each side is padded equally.
            in_channels: Number of input channels (C).
            out_features: Size of the final output vector per sample.
            softplus_beta: Beta parameter for Softplus activation.
        """
        super(Model, self).__init__()

        # Convert symmetric 3-element pad to 6-element pad expected by ReflectionPad3d:
        # ReflectionPad3d expects (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
        if not (isinstance(pad_sizes, (list, tuple)) and len(pad_sizes) == 3):
            raise ValueError("pad_sizes must be a tuple/list of three integers (pad_depth, pad_height, pad_width)")
        pad_d, pad_h, pad_w = pad_sizes
        pad6 = (pad_w, pad_w, pad_h, pad_h, pad_d, pad_d)

        self.pad = nn.ReflectionPad3d(pad6)
        self.pool = nn.AdaptiveMaxPool1d(pool_output_size)
        self.softplus = nn.Softplus(beta=softplus_beta)
        # Linear projects flattened (C * pool_output_size) to out_features
        self.fc = nn.Linear(in_channels * pool_output_size, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Steps:
            1. Reflection-pad spatial dims (D,H,W).
            2. Flatten spatial dims into a single sequence of length L = D'*H'*W'.
            3. Apply AdaptiveMaxPool1d to reduce L -> pool_output_size (on last dim).
            4. Apply Softplus activation.
            5. Flatten channels and pooled positions and apply a Linear projection.

        Args:
            x: Input tensor of shape (N, C, D, H, W)

        Returns:
            Tensor of shape (N, out_features)
        """
        # 1) Pad the 3D spatial volume
        x = self.pad(x)  # shape: (N, C, D', H', W')

        # 2) Flatten spatial dimensions into a single sequence dimension L
        N, C, Dp, Hp, Wp = x.shape
        L = Dp * Hp * Wp
        x = x.view(N, C, L)  # shape: (N, C, L)

        # 3) Adaptive max pooling along the flattened spatial axis
        x = self.pool(x)  # shape: (N, C, pool_output_size)

        # 4) Non-linear activation
        x = self.softplus(x)  # element-wise Softplus

        # 5) Flatten (C * pool_output_size) and project to output vector
        x = x.view(N, -1)  # shape: (N, C * pool_output_size)
        x = self.fc(x)     # shape: (N, out_features)

        return x

# Configuration variables (used by get_inputs / get_init_inputs)
batch_size = 8
in_channels = 3
D = 8
H = 16
W = 16

pool_output_size = 32
out_features = 128
pad_sizes = (1, 2, 1)  # (pad_depth, pad_height, pad_width)
softplus_beta = 1.0

def get_inputs() -> List[torch.Tensor]:
    """
    Create a random input tensor matching the configured shapes:
    Returns a list with a single tensor of shape (batch_size, in_channels, D, H, W).
    """
    x = torch.randn(batch_size, in_channels, D, H, W)
    return [x]

def get_init_inputs() -> List[Any]:
    """
    Returns initialization parameters for the Model constructor in the same order:
    [pool_output_size, pad_sizes, in_channels, out_features, softplus_beta]
    """
    return [pool_output_size, pad_sizes, in_channels, out_features, softplus_beta]