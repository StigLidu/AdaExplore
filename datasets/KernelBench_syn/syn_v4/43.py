import torch
import torch.nn as nn
from typing import Tuple, List

class Model(nn.Module):
    """
    Complex 3D processing module that demonstrates a mix of nonlinearities and pooling.

    Computation pattern (high level):
    1. Apply Softsign activation to the input tensor.
    2. Compute a depth-wise gating signal by averaging across the depth dimension.
    3. Modulate the activated tensor by a sigmoid gating map (channel-aware).
    4. Apply Softshrink for denoising/sparsification.
    5. Perform 3D max pooling to reduce spatial resolution.
    6. Channel-wise L1 normalization and optional scaling.

    This module accepts initialization parameters to configure the MaxPool3d kernel,
    the Softshrink lambda threshold, and a final channel scaling factor.
    """
    def __init__(self, pool_kernel: Tuple[int, int, int] = (2, 4, 4), shrink_lambda: float = 0.3, channel_scale: float = 1.5):
        """
        Args:
            pool_kernel: Kernel size (D, H, W) for MaxPool3d. Should divide input spatial dims.
            shrink_lambda: Threshold parameter for Softshrink.
            channel_scale: Multiplicative scaling applied after normalization.
        """
        super(Model, self).__init__()
        self.softsign = nn.Softsign()
        self.softshrink = nn.Softshrink(lambd=shrink_lambda)
        self.pool = nn.MaxPool3d(kernel_size=pool_kernel, stride=pool_kernel)
        self.channel_scale = float(channel_scale)
        self.eps = 1e-6  # numeric stability for normalization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, D, H, W)

        Returns:
            Tensor after activation, gating, shrinkage, pooling and normalization.
            Shape will be (B, C, D', H', W') depending on pool_kernel.
        """
        # 1) Non-linear softsign activation
        activated = self.softsign(x)  # (B, C, D, H, W)

        # 2) Depth-wise gating: mean over depth dimension to produce a (B, C, 1, H, W) map
        #    This captures coarse depth-aware context per channel and spatial location.
        gating_map = torch.mean(activated, dim=2, keepdim=True)  # (B, C, 1, H, W)

        # 3) Convert gating_map to a multiplicative gate in (0,1) via sigmoid and apply
        gate = torch.sigmoid(gating_map)
        gated = activated * gate  # (B, C, D, H, W) broadcasted along depth

        # 4) Softshrink for element-wise sparsification/denoising
        sparse = self.softshrink(gated)  # (B, C, D, H, W)

        # 5) 3D max pooling to reduce spatial resolution
        pooled = self.pool(sparse)  # (B, C, D', H', W')

        # 6) Channel-wise L1 normalization across spatial dims (D', H', W') to keep scale stable
        #    and then scale by channel_scale.
        abs_sum = torch.sum(torch.abs(pooled), dim=(2, 3, 4), keepdim=True)  # (B, C, 1, 1, 1)
        normalized = pooled / (abs_sum + self.eps)
        scaled = normalized * self.channel_scale

        return scaled


# Configuration / default inputs
batch_size = 4
channels = 8
depth = 8    # must be divisible by pool_kernel[0]
height = 64  # must be divisible by pool_kernel[1]
width = 64   # must be divisible by pool_kernel[2]

# Pool kernel chosen to evenly divide spatial dims
POOL_KERNEL = (2, 4, 4)
SHRINK_LAMBDA = 0.25
CHANNEL_SCALE = 2.0

def get_inputs() -> List[torch.Tensor]:
    """
    Prepare a random input tensor matching the configured shapes.

    Returns:
        A single-element list containing a tensor of shape (B, C, D, H, W).
    """
    x = torch.randn(batch_size, channels, depth, height, width)
    return [x]

def get_init_inputs() -> List:
    """
    Return the initialization parameters for the Model constructor.

    Returns:
        [pool_kernel, shrink_lambda, channel_scale]
    """
    return [POOL_KERNEL, SHRINK_LAMBDA, CHANNEL_SCALE]