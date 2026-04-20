import torch
import torch.nn as nn
import torch.nn.functional as F

# Configuration
batch_size = 8
in_channels = 3
mid_channels = 16
out_channels = 5
in_height = 16
in_width = 16

# ConvTranspose2d parameters
kernel_size = 4
stride = 2
padding = 1
output_padding = 0

# CELU alpha
celu_alpha = 0.1

class Model(nn.Module):
    """
    Complex example combining ConvTranspose2d, CELU activation and Softmax2d.
    The model upsamples the input with a transposed convolution, applies a non-linearity,
    refines with a second transposed convolution, adds a projected skip connection,
    applies another CELU, and finally normalizes per-spatial-location with Softmax2d.
    """
    def __init__(
        self,
        in_ch: int = in_channels,
        mid_ch: int = mid_channels,
        out_ch: int = out_channels,
        k: int = kernel_size,
        s: int = stride,
        p: int = padding,
        op: int = output_padding,
        alpha: float = celu_alpha
    ):
        super(Model, self).__init__()
        # First upsampling block
        self.up1 = nn.ConvTranspose2d(in_ch, mid_ch, kernel_size=k, stride=s, padding=p, output_padding=op)
        # Non-linear activation
        self.act1 = nn.CELU(alpha=alpha)
        # Second transposed conv to refine features (keeps spatial size)
        self.up2 = nn.ConvTranspose2d(mid_ch, out_ch, kernel_size=3, stride=1, padding=1)
        # Project input to match out_ch for residual add (1x1 conv)
        self.proj_skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1)
        # Second CELU after combining skip
        self.act2 = nn.CELU(alpha=alpha)
        # Softmax across channel dimension per spatial location
        self.spatial_softmax = nn.Softmax2d()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        1. Upsample input spatially with ConvTranspose2d.
        2. Apply CELU non-linearity.
        3. Further refine with another ConvTranspose2d to produce out_ch channels.
        4. Project original input with a 1x1 conv and add as a residual (after upsampling to match spatial size).
        5. Apply CELU again.
        6. Apply Softmax2d to obtain per-location channel distributions.
        """
        # 1. Upsample
        y = self.up1(x)                     # shape: (B, mid_ch, H*2, W*2)
        # 2. Non-linearity
        y = self.act1(y)
        # 3. Refine & map to out_ch
        y = self.up2(y)                     # shape: (B, out_ch, H*2, W*2)
        # 4. Prepare skip: project input channels and upsample spatially to match y
        skip = self.proj_skip(x)            # shape: (B, out_ch, H, W)
        skip_upsampled = F.interpolate(skip, size=y.shape[-2:], mode='nearest')  # match spatial dims
        # Combine with residual connection
        y = y + skip_upsampled
        # 5. Second non-linearity
        y = self.act2(y)
        # 6. Softmax2d across channels per spatial location
        y = self.spatial_softmax(y)
        return y

def get_inputs():
    """
    Returns a single input tensor with shape (batch_size, in_channels, in_height, in_width).
    """
    x = torch.randn(batch_size, in_channels, in_height, in_width)
    return [x]

def get_init_inputs():
    """
    Return initialization parameters for Model constructor in the following order:
    [in_ch, mid_ch, out_ch, kernel_size, stride, padding, output_padding, alpha]
    """
    return [in_channels, mid_channels, out_channels, kernel_size, stride, padding, output_padding, celu_alpha]