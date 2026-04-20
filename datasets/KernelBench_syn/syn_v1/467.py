import torch
import torch.nn as nn
import math
from typing import List

class Model(nn.Module):
    """
    Complex module that combines MaxPool2d, Unfold and SiLU in a patch-wise
    gating/aggregation pattern.

    Computation steps:
    1. Apply MaxPool2d to obtain a coarse summary (spatially reduced).
    2. Extract sliding patches from the original input using nn.Unfold.
    3. Compute a channel-wise context from the pooled tensor (global average per channel).
    4. Pass the context through a SiLU activation to create multiplicative gates.
    5. Apply the gates to each spatial patch (scaling patch vectors per-channel).
    6. Aggregate the gated patch elements (average over patch pixels) and
       reshape back to a spatial map corresponding to the unfolded patch grid.
    """
    def __init__(
        self,
        in_channels: int,
        pool_kernel: int,
        pool_stride: int,
        pool_padding: int,
        unfold_kernel: int,
        unfold_stride: int,
        unfold_padding: int,
    ):
        """
        Initialize the model components.

        Args:
            in_channels (int): Number of input channels.
            pool_kernel (int): Kernel size for MaxPool2d.
            pool_stride (int): Stride for MaxPool2d.
            pool_padding (int): Padding for MaxPool2d.
            unfold_kernel (int): Kernel size for Unfold (square).
            unfold_stride (int): Stride for Unfold.
            unfold_padding (int): Padding for Unfold.
        """
        super(Model, self).__init__()
        # Pooling used for coarse context
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride, padding=pool_padding)
        # Unfold to extract patches from original input
        self.unfold = nn.Unfold(kernel_size=unfold_kernel, stride=unfold_stride, padding=unfold_padding)
        # Activation used to create multiplicative gating from context
        self.silu = nn.SiLU()
        # Save configuration for shape computations
        self.in_channels = in_channels
        self.unfold_kernel = unfold_kernel
        self.unfold_stride = unfold_stride
        self.unfold_padding = unfold_padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (N, C, H_out, W_out) where H_out and W_out
                          correspond to the number of patch locations extracted by Unfold.
        """
        N, C, H, W = x.shape
        # 1) coarse spatial reduction
        pooled = self.pool(x)  # (N, C, H_pool, W_pool)

        # 2) channel-wise global context from pooled (mean over spatial dims)
        # produce shape (N, C, 1, 1)
        context = pooled.mean(dim=(2, 3), keepdim=True)

        # 3) gate via SiLU (element-wise)
        gate = self.silu(context)  # (N, C, 1, 1)

        # 4) extract patches from the original input
        # unfold_out shape: (N, C * K * K, L) where L is number of sliding blocks
        unfold_out = self.unfold(x)

        K = self.unfold_kernel
        # reshape to (N, C, K*K, L) to allow per-channel per-patch pixel operations
        # Note: K*K is the number of pixels in each patch per channel
        L = unfold_out.shape[-1]
        patches = unfold_out.view(N, C, K * K, L)  # (N, C, K*K, L)

        # 5) apply channel-wise gate to each patch pixel: broadcast over pixel and locations
        # gate: (N, C, 1, 1) -> expand to (N, C, 1, L) then multiply
        gated_patches = patches * gate.expand(-1, -1, 1, L)  # (N, C, K*K, L)

        # 6) aggregate pixel information inside each patch by averaging over the K*K dimension
        # result shape: (N, C, L)
        aggregated = gated_patches.mean(dim=2)  # (N, C, L)

        # 7) reshape aggregated features back into spatial layout corresponding to L
        # Compute H_out and W_out from original H, W and unfold parameters:
        H_out = (H + 2 * self.unfold_padding - K) // self.unfold_stride + 1
        W_out = (W + 2 * self.unfold_padding - K) // self.unfold_stride + 1
        # Sanity check: H_out * W_out should equal L
        # If not, fallback to flattening into (N, C, L) to avoid shape errors
        if H_out * W_out != L:
            # return flattened patch-feature map
            return aggregated
        out = aggregated.view(N, C, H_out, W_out)
        return out

# Module-level configuration variables
batch_size = 8
in_channels = 3
height = 64
width = 64

# Pool parameters
pool_kernel = 3
pool_stride = 2
pool_padding = 1

# Unfold (patch) parameters
unfold_kernel = 5
unfold_stride = 4
unfold_padding = 1

def get_inputs() -> List[torch.Tensor]:
    """
    Returns a list containing a single input tensor matching the configured shape.
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs() -> List:
    """
    Returns initialization parameters for the Model constructor in the same order.
    """
    return [
        in_channels,
        pool_kernel,
        pool_stride,
        pool_padding,
        unfold_kernel,
        unfold_stride,
        unfold_padding,
    ]