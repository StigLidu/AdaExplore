import torch
import torch.nn as nn
from typing import Tuple, List, Any

class Model(nn.Module):
    """
    A 3D pooling + unpooling model that demonstrates a non-trivial flow:
    1. FractionalMaxPool3d to reduce spatial dims (returns pooled output and indices)
    2. SELU activation on pooled tensor
    3. 1x1x1 Conv3d to mix channel information
    4. MaxUnpool3d using the indices from fractional pooling to restore spatial dims
    5. Residual fusion with the original input scaled by a learnable factor, followed by SELU

    This combines nn.FractionalMaxPool3d, nn.SELU, and nn.MaxUnpool3d in a small pipeline.
    """
    def __init__(self, in_channels: int, kernel_size: Tuple[int, int, int] = (2, 2, 2), output_ratio: Tuple[float, float, float] = (0.5, 0.5, 0.5)):
        """
        Args:
            in_channels (int): Number of channels in the input tensor.
            kernel_size (tuple): Pooling kernel size for FractionalMaxPool3d and MaxUnpool3d.
            output_ratio (tuple): Fractional output ratios for FractionalMaxPool3d.
        """
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.output_ratio = output_ratio

        # Fractional pooling with indices so we can unpool later
        self.fracpool = nn.FractionalMaxPool3d(kernel_size=self.kernel_size, output_ratio=self.output_ratio, return_indices=True)

        # 1x1x1 conv to mix channels after pooling
        self.conv1x1 = nn.Conv3d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, bias=True)

        # SELU activation used twice in the pipeline
        self.selu = nn.SELU()

        # MaxUnpool3d to invert the pooling using the indices from fractional pooling
        self.unpool = nn.MaxUnpool3d(kernel_size=self.kernel_size)

        # Learnable residual scaling to fuse original and unpooled features
        self.res_scale = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, D, H, W)

        Returns:
            torch.Tensor: Output tensor of same shape as input after pooling/unpooling + fusion
        """
        # 1) Fractional pooling -> returns (pooled, indices)
        pooled, indices = self.fracpool(x)  # pooled shape: reduced spatial dims

        # 2) Non-linear activation
        activated = self.selu(pooled)

        # 3) Channel mixing via 1x1x1 conv
        mixed = self.conv1x1(activated)

        # 4) Unpool back to original spatial dims using indices; provide output_size to ensure correct shape
        unpooled = self.unpool(mixed, indices, output_size=x.size())

        # 5) Residual fusion with learnable scaling and final SELU
        fused = self.selu(unpooled + self.res_scale * x)

        return fused

# Module-level configuration variables
batch_size = 2
channels = 8
depth = 16
height = 24
width = 24
kernel_size = (2, 2, 2)
output_ratio = (0.5, 0.5, 0.5)

def get_inputs() -> List[torch.Tensor]:
    """
    Returns a list of input tensors for the Model.forward method.
    """
    x = torch.randn(batch_size, channels, depth, height, width)
    return [x]

def get_init_inputs() -> List[Any]:
    """
    Returns initialization parameters for the Model constructor.
    """
    return [channels, kernel_size, output_ratio]