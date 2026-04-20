import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    A small 3D residual-like block that combines Conv3d, LazyBatchNorm3d, PReLU, and RReLU.
    
    The forward pass does:
      1. 3x3x3 Conv3d -> PReLU (channel-wise) -> LazyBatchNorm3d
      2. 1x1x1 Conv3d -> RReLU
      3. Residual add (with optional 1x1x1 projection if channels differ)
      4. Adaptive average pooling to a single spatial location and optional squeeze
    
    This demonstrates mixing normalization, parameterized activation, and randomized activation
    in a slightly more involved computation with multiple tensor shapes.
    """
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, kernel_size: int = 3):
        """
        Initializes the composite 3D module.

        Args:
            in_channels (int): Number of input channels.
            mid_channels (int): Number of channels in the intermediate conv.
            out_channels (int): Number of output channels.
            kernel_size (int, optional): Kernel size for the first Conv3d. Defaults to 3.
        """
        super(Model, self).__init__()
        padding = kernel_size // 2

        # First convolution expands or transforms channel dimension
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=True)
        # Channel-wise learnable parametric ReLU
        self.prelu = nn.PReLU(num_parameters=mid_channels)
        # Lazy BatchNorm3d will infer num_features on the first forward pass
        self.bn = nn.LazyBatchNorm3d()
        # Second convolution reduces to out_channels with 1x1x1 kernel
        self.conv2 = nn.Conv3d(mid_channels, out_channels, kernel_size=1, padding=0, bias=True)
        # Randomized leaky ReLU for a stochastic activation behavior during training
        self.rrelu = nn.RReLU(lower=0.125, upper=0.333, inplace=False)

        # If input and output channel counts differ, project the input for the residual add
        if in_channels != out_channels:
            self.proj = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
        else:
            self.proj = None

        # Small dropout to add some regularization in the residual branch
        self.drop = nn.Dropout3d(p=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, 1, 1, 1)
                          after adaptive pooling (keeps dims for convenience).
        """
        # Main path: conv -> PReLU -> BatchNorm -> conv -> RReLU
        out = self.conv1(x)
        out = self.prelu(out)
        out = self.bn(out)  # LazyBatchNorm3d will initialize num_features from out.shape[1]
        out = self.conv2(out)
        out = self.rrelu(out)

        # Optional projection for the residual connection
        res = x
        if self.proj is not None:
            res = self.proj(res)

        # Dropout on the residual branch to slightly perturb training
        res = self.drop(res)

        # Residual add and global pooling
        out = out + res
        out = F.adaptive_avg_pool3d(out, output_size=1)  # (B, C, 1, 1, 1)

        return out

# Configuration / default sizes
batch_size = 8
in_channels = 4
mid_channels = 12
out_channels = 10
depth = 16
height = 16
width = 16
kernel_size = 3

def get_inputs():
    """
    Creates a random input tensor matching the declared configuration.

    Returns:
        list: single-element list containing the input tensor.
    """
    x = torch.randn(batch_size, in_channels, depth, height, width, dtype=torch.float32)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters that should be passed to Model's constructor.

    Returns:
        list: [in_channels, mid_channels, out_channels, kernel_size]
    """
    return [in_channels, mid_channels, out_channels, kernel_size]