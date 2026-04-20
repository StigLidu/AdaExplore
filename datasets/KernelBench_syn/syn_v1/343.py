import torch
import torch.nn as nn
from typing import List

class Model(nn.Module):
    """
    A composite convolutional module that demonstrates a multi-stage
    processing pipeline:

    1. 2D convolution to extract spatial features.
    2. Soft-shrink nonlinearity for sparse activations.
    3. 1x1 convolution to mix channel information.
    4. Reshape spatial dimensions into a 1D sequence per channel.
    5. Adaptive 1D average pooling to produce a fixed-length descriptor.
    6. Final channel-aggregation (mean) to produce per-sample descriptors.

    This structure combines nn.Conv2d, nn.Softshrink, and nn.AdaptiveAvgPool1d
    into a functionally distinct computation pattern compared to the examples.
    """
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_size: int,
        pool_output_size: int,
        softshrink_lambda: float = 0.5
    ):
        """
        Initializes the model with convolutional and pooling parameters.

        Args:
            in_channels (int): Number of input channels.
            mid_channels (int): Number of channels produced by the first conv.
            out_channels (int): Number of channels produced by the second conv.
            kernel_size (int): Kernel size for the first convolution (square).
            pool_output_size (int): Output length for AdaptiveAvgPool1d.
            softshrink_lambda (float): Lambda parameter for Softshrink.
        """
        super(Model, self).__init__()
        # First conv: captures local spatial context; keep same spatial size via padding
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding)
        # Soft shrinkage nonlinearity introduces sparsity
        self.softshrink = nn.Softshrink(lambd=softshrink_lambda)
        # 1x1 conv to mix channel information without changing spatial resolution
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        # Adaptive average pool over the flattened spatial sequence per channel
        self.adaptive_pool = nn.AdaptiveAvgPool1d(pool_output_size)
        # small epsilon to avoid division by zero if used later
        self.eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the module.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (N, pool_output_size) where
                          spatial information has been pooled and channels aggregated.
        """
        # Stage 1: spatial conv
        x = self.conv1(x)  # (N, mid_channels, H, W)
        # Stage 2: non-linear sparse activation
        x = self.softshrink(x)  # same shape
        # Stage 3: channel mixing
        x = self.conv2(x)  # (N, out_channels, H, W)

        # Stage 4: reshape spatial dims into a single sequence dimension L = H * W
        n, c, h, w = x.shape
        # flatten H and W into one sequence dimension
        x = x.view(n, c, h * w)  # (N, C_out, L)

        # Stage 5: adaptive average pooling to fixed-length descriptor per channel
        x = self.adaptive_pool(x)  # (N, C_out, pool_output_size)

        # Stage 6: aggregate across channels to produce final descriptor per sample
        # compute mean across channels -> (N, pool_output_size)
        out = x.mean(dim=1)  # aggregate channel-wise information

        return out

# Configuration / default instantiation parameters
batch_size = 8
in_channels = 3
height = 32
width = 32

# conv/pool parameters used for initialization in get_init_inputs()
mid_channels = 16
out_channels = 32
kernel_size = 3
pool_output_size = 8
softshrink_lambda = 0.2

def get_inputs() -> List[torch.Tensor]:
    """
    Generates a list containing a single input tensor appropriate for the model.

    Returns:
        List[torch.Tensor]: [x] where x has shape (batch_size, in_channels, height, width)
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs() -> List:
    """
    Returns initialization inputs for constructing the Model.

    The order matches the Model __init__ signature:
      in_channels, mid_channels, out_channels, kernel_size, pool_output_size, softshrink_lambda
    """
    return [in_channels, mid_channels, out_channels, kernel_size, pool_output_size, softshrink_lambda]