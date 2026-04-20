import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex 2D feature processing module that demonstrates:
    - Upsampling the input spatial resolution
    - Two parallel convolutional pathways (1x1 and 3x3)
    - A gating mechanism using nn.LogSigmoid (converted back to sigmoid via exp)
    - Lazy instance normalization (nn.LazyInstanceNorm2d) which initializes on first forward
    The resulting tensor keeps the upsampled spatial resolution and the intermediate channel dimension.
    """
    def __init__(self, in_channels: int, mid_channels: int, scale_factor: int = 2, up_mode: str = "bilinear"):
        """
        Args:
            in_channels (int): Number of input channels.
            mid_channels (int): Number of channels for intermediate feature maps.
            scale_factor (int): Spatial upsampling factor.
            up_mode (str): Upsampling mode for nn.Upsample.
        """
        super(Model, self).__init__()
        # Upsample layer - increases spatial dimensions
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=up_mode, align_corners=False if up_mode in ("bilinear", "trilinear") else None)
        # Two parallel conv branches after upsampling
        self.conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv3x3 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1, bias=True)
        # LogSigmoid for a numerically stable log-sigmoid; we'll convert back to sigmoid for gating
        self.logsigmoid = nn.LogSigmoid()
        # Lazy InstanceNorm will infer num_features from the first forward pass (mid_channels)
        self.linst = nn.LazyInstanceNorm2d()
        # A small scalar parameter to modulate gating strength (learnable)
        self.gate_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
         1. Upsample input spatially
         2. Compute two feature maps via conv1x1 and conv3x3
         3. Produce a gate from conv3x3 outputs via LogSigmoid -> exp (sigmoid)
         4. Multiply conv1x1 features by the gate (element-wise)
         5. Apply LazyInstanceNorm2d
        Returns:
            Tensor of shape (B, mid_channels, H*scale, W*scale)
        """
        # 1. Upsample
        x_up = self.upsample(x)

        # 2. Two parallel convolutions
        feat_a = self.conv1x1(x_up)       # shape: (B, mid, H*, W*)
        feat_b = self.conv3x3(x_up)       # shape: (B, mid, H*, W*)

        # 3. Gating: logsigmoid -> exp to recover sigmoid, scaled by learnable parameter
        # logsigmoid returns log(sigmoid(x)); exp gives sigmoid(x)
        gate = torch.exp(self.logsigmoid(feat_b * self.gate_scale))

        # 4. Element-wise modulation
        gated = feat_a * gate

        # 5. Normalize across channels per-instance (num_features inferred lazily)
        out = self.linst(gated)

        return out

# Configuration variables (module-level)
batch_size = 8
in_channels = 3
mid_channels = 64
height = 64
width = 64
scale_factor = 2
upsample_mode = "bilinear"

def get_inputs():
    """
    Returns a list with a single input tensor matching the configuration:
    Shape: (batch_size, in_channels, height, width)
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for the Model constructor:
    [in_channels, mid_channels, scale_factor, upsample_mode]
    """
    return [in_channels, mid_channels, scale_factor, upsample_mode]