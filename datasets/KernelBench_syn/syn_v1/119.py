import torch
import torch.nn as nn
from typing import List, Any

# Configuration
BATCH = 4
IN_CHANNELS = 8
DEPTH = 10
HEIGHT = 32
WIDTH = 32

EXPAND_FACTOR = 2        # how many channels after the first conv
THRESHOLD1 = 0.1         # first threshold: values <= THRESHOLD1 replaced with 0.0
THRESHOLD2 = -0.5        # second threshold: values <= THRESHOLD2 replaced with THRESHOLD2
POOL_KERNEL = 2          # kernel/stride for maxpool and corresponding unpool

class Model(nn.Module):
    """
    3D processing module that demonstrates a small encoder-decoder pattern using:
      - ReplicationPad3d to ensure conv preserves spatial dimensions
      - Conv3d to expand channel dimensionality
      - Threshold activation (nn.Threshold)
      - MaxPool3d with indices and MaxUnpool3d to invert the spatial pooling
      - Another Threshold and a 1x1x1 Conv3d to project back to input channels

    The forward pass:
      x -> ReplicationPad3d -> Conv3d -> Threshold1 -> MaxPool3d(return_indices=True)
        -> MaxUnpool3d(using indices and saved output_size) -> Threshold2 -> Conv3d(1x1x1) -> out
    """
    def __init__(
        self,
        in_channels: int,
        expand: int = EXPAND_FACTOR,
        threshold1: float = THRESHOLD1,
        threshold2: float = THRESHOLD2,
        pool_kernel: int = POOL_KERNEL
    ):
        super(Model, self).__init__()
        hidden_channels = in_channels * expand

        # Pad so that a 3x3x3 conv with no additional padding produces the same spatial dims
        self.pad = nn.ReplicationPad3d(1)

        # Expand channels with a 3x3x3 conv (no internal padding because pad handles it)
        self.expand_conv = nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=0, bias=True)

        # Element-wise threshold activation: values <= threshold replaced with "value"
        self.threshold1 = nn.Threshold(threshold1, 0.0)

        # Pooling with indices so we can unpool later
        self.pool = nn.MaxPool3d(kernel_size=pool_kernel, stride=pool_kernel, return_indices=True)

        # Unpool layer to approximately invert the pooling
        self.unpool = nn.MaxUnpool3d(kernel_size=pool_kernel, stride=pool_kernel)

        # Second threshold to clamp very negative values (acts like a floor)
        self.threshold2 = nn.Threshold(threshold2, threshold2)

        # Project channels back to input channels with a 1x1x1 conv
        self.project_conv = nn.Conv3d(hidden_channels, in_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, D, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (N, C, D, H, W)
        """
        # Replicate-pad boundaries so conv keeps spatial dims
        padded = self.pad(x)  # (N, C, D+2, H+2, W+2)

        # Expand channels while preserving spatial dims
        conv_out = self.expand_conv(padded)  # (N, C*expand, D, H, W)

        # Apply a mild sparsifying threshold (clamp small values to zero)
        thr_out = self.threshold1(conv_out)

        # Save size for unpool reconstruction
        size_before_pool = thr_out.size()

        # MaxPool with indices for unpooling later
        pooled, indices = self.pool(thr_out)  # pooled shape reduced by pool_kernel on D,H,W

        # Unpool back to previous spatial dimensions using indices and saved size
        unpooled = self.unpool(pooled, indices, output_size=size_before_pool)

        # Apply a second threshold that floors very negative activations
        clamped = self.threshold2(unpooled)

        # Project channels back to original input channels
        out = self.project_conv(clamped)

        return out

def get_inputs() -> List[torch.Tensor]:
    """
    Generate example inputs for the module.

    Returns:
        List[torch.Tensor]: [x] where x has shape (BATCH, IN_CHANNELS, DEPTH, HEIGHT, WIDTH)
    """
    x = torch.randn(BATCH, IN_CHANNELS, DEPTH, HEIGHT, WIDTH)
    return [x]

def get_init_inputs() -> List[Any]:
    """
    Return initialization parameters suitable for constructing the Model.

    Returns:
        List[Any]: [in_channels, expand, threshold1, threshold2, pool_kernel]
    """
    return [IN_CHANNELS, EXPAND_FACTOR, THRESHOLD1, THRESHOLD2, POOL_KERNEL]