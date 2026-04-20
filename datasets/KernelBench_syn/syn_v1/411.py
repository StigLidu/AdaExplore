import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    A moderately complex 3D vision-style model combining Conv3d, LocalResponseNorm and LazyConv3d.
    The model performs an initial 3D convolution to project input channels, applies ReLU and local
    response normalization, then uses a lazily-initialized 3D convolution to produce an output
    channel projection. Spatial dimensions are aggregated with AdaptiveAvgPool3d and the result
    is passed through a fully-connected layer producing a compact feature vector.
    """
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        lrn_size: int = 5,
        fc_out: int = 128
    ):
        """
        Args:
            in_channels (int): Number of channels in the input tensor.
            mid_channels (int): Number of channels produced by the first Conv3d.
            out_channels (int): Number of channels produced by the LazyConv3d.
            kernel_size (int, optional): Kernel size for the first Conv3d. Defaults to 3.
            stride (int, optional): Stride for the first Conv3d. Defaults to 1.
            padding (int, optional): Padding for the first Conv3d. Defaults to 1.
            lrn_size (int, optional): Size parameter for LocalResponseNorm. Defaults to 5.
            fc_out (int, optional): Output dimension of the final fully-connected layer. Defaults to 128.
        """
        super(Model, self).__init__()
        # First explicit 3D convolution to establish an initial feature representation
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        # Local Response Normalization to introduce a normalization that emphasizes large activations
        self.lrn = nn.LocalResponseNorm(lrn_size, alpha=1e-4, beta=0.75, k=2.0)
        # LazyConv3d will infer its in_channels at the first forward call from the previous layer output
        # Using stride=2 to reduce spatial resolution and make the model do a non-trivial spatial transform
        self.lazy_conv = nn.LazyConv3d(out_channels, kernel_size=3, stride=2, padding=1)
        # Global aggregation over the remaining spatial dimensions to a 1x1x1 volume per channel
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # Final linear layer to produce a compact feature vector
        self.fc = nn.Linear(out_channels, fc_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation:
          1. conv1 -> ReLU
          2. local response normalization
          3. lazy_conv -> ReLU
          4. adaptive avg pooling to 1x1x1
          5. flatten and fully connected + sigmoid activation

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, fc_out) with values in (0,1)
        """
        x = self.conv1(x)
        x = F.relu(x, inplace=True)
        x = self.lrn(x)
        x = self.lazy_conv(x)  # initializes its weight/bias lazily based on conv1 output channels
        x = F.relu(x, inplace=True)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x

# Configuration variables for test inputs
batch_size = 8
in_channels = 3
depth = 16
height = 64
width = 64
mid_channels = 16
out_channels = 32
kernel_size = 3
stride = 1
padding = 1
lrn_size = 5
fc_out = 128

def get_inputs():
    """
    Returns a list containing a single input tensor sized according to the configuration variables.
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Returns the list of initialization arguments for the Model constructor in the same order.
    """
    return [in_channels, mid_channels, out_channels, kernel_size, stride, padding, lrn_size, fc_out]