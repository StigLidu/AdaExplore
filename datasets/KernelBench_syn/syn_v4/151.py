import torch
import torch.nn as nn

# Configuration
batch_size = 8
channels = 16
height = 32
width = 32

class Model(nn.Module):
    """
    A slightly more complex module that combines Instance Normalization,
    a Tanhshrink non-linearity, learnable per-channel affine scaling,
    and spatial downsampling via MaxPool2d. After pooling it computes
    a spatial summary and applies a residual modulation.

    Computation graph (high-level):
      1. InstanceNorm2d -> normalization over channels per-instance
      2. Tanhshrink -> elementwise non-linearity (x - tanh(x))
      3. Per-channel learnable scale and bias (broadcasted)
      4. MaxPool2d (2x2 stride 2) -> spatial downsampling
      5. Spatial mean -> (N, C, 1, 1)
      6. Tanh of mean added back as a residual (broadcasted)
    """
    def __init__(self, num_channels: int):
        super(Model, self).__init__()
        # Instance normalization over channels (no affine: we'll apply our own)
        self.inst_norm = nn.InstanceNorm2d(num_channels, affine=False, eps=1e-5, momentum=0.1)
        # Element-wise Tanhshrink non-linearity
        self.tanhshrink = nn.Tanhshrink()
        # Max pooling to reduce spatial dimensions by factor of 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Learnable per-channel scale and bias (broadcastable to (N, C, H, W))
        self.scale = nn.Parameter(torch.ones(num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(num_channels, 1, 1))

        # Initialize parameters more explicitly (keeps clarity similar to nn Modules)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.scale)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
          x: Tensor of shape (N, C, H, W)

        Returns:
          Tensor of shape (N, C, H/2, W/2) after the described operations.
        """
        # 1. Instance normalization
        normalized = self.inst_norm(x)

        # 2. Tanhshrink activation (element-wise)
        activated = self.tanhshrink(normalized)

        # 3. Per-channel affine transform (broadcast scale and bias)
        # activated has shape (N, C, H, W); scale/bias are (C,1,1)
        scaled = activated * self.scale + self.bias

        # 4. Spatial downsampling
        pooled = self.pool(scaled)

        # 5. Spatial mean (global per-channel summary)
        spatial_mean = pooled.mean(dim=(2, 3), keepdim=True)  # shape (N, C, 1, 1)

        # 6. Residual modulation: add tanh(spatial_mean) back to pooled
        out = pooled + torch.tanh(spatial_mean)

        return out

def get_inputs():
    """
    Create a random input tensor consistent with the module-level configuration.

    Returns:
        list: [x] where x is a torch.Tensor shaped (batch_size, channels, height, width)
    """
    x = torch.randn(batch_size, channels, height, width)
    return [x]

def get_init_inputs():
    """
    Return initialization parameters for the Model.

    Returns:
        list: [channels] to be used when constructing Model(channels)
    """
    return [channels]