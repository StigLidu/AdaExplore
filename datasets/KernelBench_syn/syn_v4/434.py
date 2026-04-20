import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex 2D feature re-scaler and gated activation module.

    Pipeline:
    1. Reflection-pad the input to enlarge spatial boundaries.
    2. Channel-wise scale (learnable) applied to the padded input.
    3. Channel-wise Softmax2d to produce spatially-varying channel gates.
    4. Element-wise gating: scaled * gate.
    5. Add an external scalar bias (broadcast) and apply ReLU.

    This creates a combined normalization/gating pattern that is different from
    plain activations or pooling layers.
    """
    def __init__(self, in_channels: int, padding=(1, 1, 1, 1), relu_inplace: bool = False, scale_init: float = 1.0):
        """
        Args:
            in_channels (int): Number of channels expected in the input (C).
            padding (int or tuple): Padding argument passed to ReflectionPad2d.
            relu_inplace (bool): Whether ReLU should be inplace.
            scale_init (float): Initial value for the per-channel scale parameter.
        """
        super(Model, self).__init__()
        # Reflection padding to preserve edge information while expanding the spatial grid
        self.pad = nn.ReflectionPad2d(padding)
        # Per-channel learnable scale parameter (C,)
        self.scale = nn.Parameter(torch.full((in_channels,), float(scale_init)))
        # ReLU non-linearity
        self.relu = nn.ReLU(inplace=relu_inplace)
        # Softmax across channels for each spatial location
        self.softmax2d = nn.Softmax2d()

    def forward(self, x: torch.Tensor, bias: float) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W)
            bias (float): External scalar added before ReLU (broadcast across all dims)

        Returns:
            torch.Tensor: Output tensor after padding, scaling, gating and activation.
        """
        # 1) Reflection pad
        x_padded = self.pad(x)

        # 2) Channel-wise scaling: expand scale to (1, C, 1, 1) for broadcasting
        C = x_padded.shape[1]
        scale_view = self.scale.view(1, C, 1, 1)
        scaled = x_padded * scale_view

        # 3) Compute channel gates via Softmax2d (over channels for each spatial location)
        gates = self.softmax2d(scaled)

        # 4) Apply gating
        gated = scaled * gates

        # 5) Add scalar bias (broadcast) and apply ReLU
        out = self.relu(gated + bias)

        return out

# Module-level configuration variables
batch_size = 8
channels = 16
height = 32
width = 24
padding = (2, 2, 1, 1)  # (left, right, top, bottom) reflection padding
relu_inplace = False
scale_init = 0.75

def get_inputs():
    """
    Returns typical runtime inputs:
    - x: random tensor of shape (batch_size, channels, height, width)
    - bias: small scalar to bias activations before ReLU
    """
    x = torch.randn(batch_size, channels, height, width)
    bias = 0.1
    return [x, bias]

def get_init_inputs():
    """
    Returns initialization arguments for Model constructor:
    [in_channels, padding, relu_inplace, scale_init]
    """
    return [channels, padding, relu_inplace, scale_init]