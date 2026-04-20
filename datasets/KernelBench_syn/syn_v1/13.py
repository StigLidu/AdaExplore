import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex 2D preprocessing module that:
      1. Applies reflection padding to preserve boundary information.
      2. Performs 2D average pooling to reduce spatial resolution.
      3. Spatially normalizes each channel (zero-mean, unit-variance with epsilon).
      4. Applies LogSoftmax across the channel dimension so each spatial location's channels
         form a log-probability distribution.

    This combination is useful for creating a stabilized, spatially-aware per-location
    categorical (log-probability) representation from dense feature maps.
    """
    def __init__(self, padding: tuple, pool_kernel: int, pool_stride: int = None, eps: float = 1e-5):
        """
        Args:
            padding (tuple): ReflectionPad2d padding as (left, right, top, bottom).
            pool_kernel (int): Kernel size for AvgPool2d.
            pool_stride (int, optional): Stride for AvgPool2d. If None, stride == pool_kernel.
            eps (float, optional): Small constant to avoid division by zero in standardization.
        """
        super(Model, self).__init__()
        self.pad = nn.ReflectionPad2d(padding)
        self.pool = nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_stride)
        # LogSoftmax will be applied over the channel dimension (dim=1)
        self.logsoft = nn.LogSoftmax(dim=1)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the composite module.

        Steps:
            x -> ReflectionPad2d -> AvgPool2d -> channel-wise spatial normalization ->
                 LogSoftmax over channels

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (N, C, H_out, W_out) where the values
                          are log-probabilities across channels for each spatial location.
        """
        # 1) Preserve boundaries with reflection padding
        x = self.pad(x)

        # 2) Downsample spatial dimensions via average pooling
        x = self.pool(x)

        # 3) Spatial normalization per-channel: make each channel zero-mean and unit-variance
        #    Compute mean and std across spatial dims (H, W)
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True, unbiased=False)
        x = (x - mean) / (std + self.eps)

        # 4) Convert per-location channel activations into log-probabilities over channels
        out = self.logsoft(x)
        return out

# Configuration / default initialization parameters
batch_size = 8
channels = 16
height = 32
width = 48
# ReflectionPad2d expects (left, right, top, bottom)
padding = (2, 2, 1, 1)
pool_kernel = 3
pool_stride = 2
eps = 1e-6

def get_inputs():
    """
    Returns a list with a single input tensor matching (batch_size, channels, height, width).
    """
    x = torch.randn(batch_size, channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters used to construct the Model:
    [padding, pool_kernel, pool_stride, eps]
    """
    return [padding, pool_kernel, pool_stride, eps]