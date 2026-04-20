import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex vision-processing module that:
    - Uses PixelUnshuffle to downsample spatial resolution while increasing channels.
    - Applies a 1x1 convolution to mix the expanded channels.
    - Uses GELU nonlinearity.
    - Applies a depthwise 3x3 convolution for spatial filtering.
    - Performs global average pooling and a final linear projection to output feature logits.
    - Applies LogSoftmax to produce log-probabilities across output features.
    """
    def __init__(self, in_channels: int, downscale_factor: int, mid_channels: int, out_features: int):
        """
        Initializes the composite model.

        Args:
            in_channels (int): Number of input channels (e.g., 3 for RGB).
            downscale_factor (int): Factor by which height and width are downscaled by PixelUnshuffle.
            mid_channels (int): Number of channels after the first 1x1 convolution.
            out_features (int): Number of output features/logits.
        """
        super(Model, self).__init__()
        assert downscale_factor >= 1 and isinstance(downscale_factor, int), "downscale_factor must be an int >= 1"

        self.downscale = nn.PixelUnshuffle(downscale_factor)
        # After PixelUnshuffle, channels become in_channels * (downscale_factor ** 2)
        expanded_channels = in_channels * (downscale_factor ** 2)
        self.conv1x1 = nn.Conv2d(expanded_channels, mid_channels, kernel_size=1, bias=True)
        self.gelu = nn.GELU()
        # Depthwise convolution: groups == mid_channels
        self.depthwise = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels, bias=True)
        # Projection from pooled channels to output features
        self.proj = nn.Linear(mid_channels, out_features, bias=True)
        # LogSoftmax across feature dimension
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the composite module.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W), where H and W are divisible by downscale_factor.

        Returns:
            torch.Tensor: Log-probabilities of shape (B, out_features).
        """
        # Step 1: PixelUnshuffle to reduce spatial dims and increase channels
        x = self.downscale(x)

        # Step 2: 1x1 conv to mix the expanded channels
        x = self.conv1x1(x)

        # Step 3: Non-linear activation
        x = self.gelu(x)

        # Step 4: Depthwise spatial filtering
        x = self.depthwise(x)

        # Step 5: Global average pooling over spatial dimensions -> (B, mid_channels)
        x = x.mean(dim=(2, 3))  # adaptive average pooling equivalent

        # Step 6: Linear projection to output features
        x = self.proj(x)

        # Step 7: LogSoftmax across feature dimension
        out = self.logsoftmax(x)
        return out

# Configuration / default parameters
batch_size = 8
in_channels = 3
downscale_factor = 2  # H and W must be divisible by this
mid_channels = 64
out_features = 100
height = 128  # divisible by downscale_factor
width = 128   # divisible by downscale_factor

def get_inputs():
    """
    Returns the input tensors for running the model.
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters for the Model constructor.
    """
    return [in_channels, downscale_factor, mid_channels, out_features]