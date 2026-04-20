import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Any

class Model(nn.Module):
    """
    Complex model combining Threshold, MaxPool3d/MaxUnpool3d, and Softmax2d.

    Pipeline:
      1. Apply element-wise Threshold activation.
      2. Perform 3D max-pooling (to obtain pooled values and indices).
      3. Affine-scale the pooled tensor (learnable scale and bias).
      4. Use MaxUnpool3d with the stored indices to reconstruct spatial-temporal volume.
      5. Collapse the depth dimension (mean over D).
      6. Apply Softmax2d to produce per-location channel probabilities.

    Input shape: (N, C, D, H, W)
    Output shape: (N, C, H, W) -- Softmaxed across channels at each spatial location.
    """
    def __init__(
        self,
        kernel_size: int,
        stride: int = None,
        padding: int = 0,
        threshold: float = 0.0,
        replace_value: float = 0.0,
        scale_init: float = 1.0,
        bias_init: float = 0.0,
        channels: int = 8,
    ):
        """
        Args:
            kernel_size (int): kernel for MaxPool3d and MaxUnpool3d.
            stride (int, optional): stride for pooling/unpooling. Defaults to kernel_size if None.
            padding (int, optional): padding for pooling/unpooling. Defaults to 0.
            threshold (float): threshold value for nn.Threshold (values <= threshold replaced).
            replace_value (float): value to replace elements <= threshold.
            scale_init (float): initial value for the learnable scale parameter.
            bias_init (float): initial value for the learnable bias parameter.
            channels (int): number of channels C in the input tensor.
        """
        super(Model, self).__init__()
        if stride is None:
            stride = kernel_size

        self.pool_kernel = kernel_size
        self.pool_stride = stride
        self.pool_padding = padding
        self.channels = channels

        # Non-linear thresholding
        self.threshold = nn.Threshold(threshold, replace_value)

        # MaxUnpool3d layer to invert pooling using indices
        self.unpool = nn.MaxUnpool3d(kernel_size=kernel_size, stride=stride, padding=padding)

        # Softmax2d to normalize across channels per spatial location (H, W)
        self.softmax2d = nn.Softmax2d()

        # Learnable affine transform applied to pooled features before unpooling
        # Shape: (1, C, 1, 1, 1) to broadcast over N, D, H, W appropriately
        scale_tensor = torch.full((1, channels, 1, 1, 1), float(scale_init), dtype=torch.float32)
        bias_tensor = torch.full((1, channels, 1, 1, 1), float(bias_init), dtype=torch.float32)
        self.scale = nn.Parameter(scale_tensor)
        self.bias = nn.Parameter(bias_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation.

        Args:
            x: Input tensor of shape (N, C, D, H, W)

        Returns:
            Tensor of shape (N, C, H, W) after unpooling collapse and Softmax2d.
        """
        # 1) Threshold activation
        x_thresh = self.threshold(x)

        # 2) MaxPool3d with indices (functional API provides indices)
        pooled, indices = F.max_pool3d(
            x_thresh,
            kernel_size=self.pool_kernel,
            stride=self.pool_stride,
            padding=self.pool_padding,
            return_indices=True,
        )

        # 3) Affine scaling (broadcasting over spatial dims)
        scaled = pooled * self.scale + self.bias

        # 4) Unpool back to original volume size using indices
        # Provide output_size to ensure exact reconstruction shape
        unpooled = self.unpool(scaled, indices, output_size=x.size())

        # 5) Collapse depth (D) dimension by mean to obtain (N, C, H, W)
        collapsed = unpooled.mean(dim=2)

        # 6) Apply Softmax2d across channels for each (H, W) location
        out = self.softmax2d(collapsed)

        return out

# Module-level configuration variables
batch_size = 4
channels = 8
depth = 16
height = 32
width = 32

kernel_size = 2
stride = 2
padding = 0

threshold_val = 0.05
replace_value = -0.1
scale_init = 1.25
bias_init = 0.01

def get_inputs() -> List[torch.Tensor]:
    """
    Returns inputs for a forward pass:
      - x: random tensor shaped (batch_size, channels, depth, height, width)
    """
    x = torch.randn(batch_size, channels, depth, height, width, dtype=torch.float32)
    return [x]

def get_init_inputs() -> List[Any]:
    """
    Returns the initialization parameters to construct Model(...).
    The ordering matches Model.__init__ arguments.
    """
    return [kernel_size, stride, padding, threshold_val, replace_value, scale_init, bias_init, channels]