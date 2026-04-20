import torch
import torch.nn as nn

# Configuration variables
BATCH = 2
C_IN = 3
MID_CH = 16
OUT_CH = 8
KERNEL_SIZE = 3
POOL_KERNEL = 2
NEGATIVE_SLOPE = 0.1
CELU_ALPHA = 1.0
DEPTH = 16
HEIGHT = 16
WIDTH = 16

class Model(nn.Module):
    """
    Complex 3D processing module that demonstrates a small encoder-like pattern:
    - 3D convolution
    - LeakyReLU activation
    - MaxPool3d with indices (for later unpooling)
    - MaxUnpool3d to restore spatial resolution
    - 1x1x1 convolution projection
    - CELU activation
    
    The pattern intentionally preserves channel consistency between pooling and unpooling
    (i.e., no channel-changing operations between pool and unpool).
    """
    def __init__(
        self,
        in_channels: int = C_IN,
        mid_channels: int = MID_CH,
        out_channels: int = OUT_CH,
        kernel_size: int = KERNEL_SIZE,
        pool_kernel: int = POOL_KERNEL,
        negative_slope: float = NEGATIVE_SLOPE,
        celu_alpha: float = CELU_ALPHA,
    ):
        super(Model, self).__init__()
        # Encoder conv that preserves spatial dims with padding
        padding = kernel_size // 2
        self.conv = nn.Conv3d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding)
        # Non-linear activation after conv
        self.act1 = nn.LeakyReLU(negative_slope=negative_slope, inplace=False)
        # Pooling layer that returns indices for unpooling
        self.pool = nn.MaxPool3d(kernel_size=pool_kernel, stride=pool_kernel, return_indices=True)
        # Unpooling layer that uses the indices to invert the pooling
        self.unpool = nn.MaxUnpool3d(kernel_size=pool_kernel, stride=pool_kernel)
        # Projection after unpooling (channel change)
        self.proj = nn.Conv3d(mid_channels, out_channels, kernel_size=1)
        # Final non-linearity
        self.act2 = nn.CELU(alpha=celu_alpha, inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        1) 3D convolution -> (B, mid, D, H, W)
        2) LeakyReLU
        3) MaxPool3d(return_indices=True) -> (B, mid, D/2, H/2, W/2), indices
        4) MaxUnpool3d(pooled, indices, output_size=pre_pool.shape) -> (B, mid, D, H, W)
        5) 1x1x1 Conv to change channels -> (B, out, D, H, W)
        6) CELU activation -> output
        """
        x_conv = self.conv(x)
        x_act = self.act1(x_conv)
        pooled, indices = self.pool(x_act)
        # Provide output_size to unpool so spatial dims are restored correctly
        unpooled = self.unpool(pooled, indices, output_size=x_act.size())
        projected = self.proj(unpooled)
        out = self.act2(projected)
        return out

def get_inputs():
    """
    Create a 5D input tensor for the 3D model: (BATCH, C_IN, DEPTH, HEIGHT, WIDTH)
    """
    x = torch.randn(BATCH, C_IN, DEPTH, HEIGHT, WIDTH)
    return [x]

def get_init_inputs():
    """
    Initialization parameters for Model constructor so external test harnesses
    can instantiate it with the same configuration used to create inputs.
    """
    return [C_IN, MID_CH, OUT_CH, KERNEL_SIZE, POOL_KERNEL, NEGATIVE_SLOPE, CELU_ALPHA]