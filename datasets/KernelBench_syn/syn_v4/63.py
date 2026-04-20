import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A 3D residual block that combines convolutional transforms with lazy instance normalization
    and multiple nonlinearities. The module demonstrates:
      - A 3D convolution to increase feature dimensionality
      - LazyInstanceNorm3d to lazily infer num_features from the incoming tensor
      - PReLU activation (channel-wise)
      - A 1x1 projection for the residual path
      - Softsign on the residual branch and an element-wise residual addition

    The design ensures the normalization layer is initialized lazily from the first forward pass.
    """
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int):
        """
        Args:
            in_channels (int): Number of channels in the input tensor.
            mid_channels (int): Number of channels after the first convolution.
            out_channels (int): Number of channels in the output tensor (residual addition target).
        """
        super(Model, self).__init__()
        # First 3D convolution expands/changes feature dimensionality
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, padding=1)
        # Lazy instance normalization: num_features will be set on first forward based on conv1 output
        self.norm = nn.LazyInstanceNorm3d()
        # Channel-wise learnable PReLU; one parameter per channel for expressivity
        self.prelu = nn.PReLU(num_parameters=mid_channels)
        # Second 3D convolution that projects to desired output channels
        self.conv2 = nn.Conv3d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1)
        # 1x1 convolution to project the input to the same shape as conv2 output for residual addition
        self.res_proj = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        # Softsign activation applied to the residual branch before addition
        self.softsign = nn.Softsign()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the composite block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth, height, width).
        """
        # Primary path: conv -> instance norm (lazy) -> PReLU -> conv
        y = self.conv1(x)          # convolution expands channels
        y = self.norm(y)           # lazy instance normalization (num_features inferred on first call)
        y = self.prelu(y)          # learnable nonlinearity
        y = self.conv2(y)          # project to output channels

        # Residual path: 1x1 projection -> softsign nonlinearity
        res = self.res_proj(x)
        res = self.softsign(res)

        # Element-wise addition combining both paths
        out = y + res

        return out

# Configuration / shape parameters
batch_size = 2
in_channels = 16
mid_channels = 32
out_channels = 16  # chosen to allow residual addition back into original channel dimension
depth = 8
height = 32
width = 32

def get_inputs():
    """
    Creates and returns the runtime input tensors for the model.

    Returns:
        list: [x] where x is a torch.Tensor of shape (batch_size, in_channels, depth, height, width)
    """
    x = torch.randn(batch_size, in_channels, depth, height, width, dtype=torch.float32)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters required to construct the Model instance.

    Returns:
        list: [in_channels, mid_channels, out_channels]
    """
    return [in_channels, mid_channels, out_channels]