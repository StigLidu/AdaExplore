import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex convolutional block that demonstrates:
    - a primary convolution followed by SiLU activation,
    - a 1x1 convolutional shortcut (residual-style addition),
    - lazy InstanceNorm2d that is initialized on first forward,
    - element-wise Threshold activation to clip small activations,
    - global spatial averaging and a final linear projection.

    This module intentionally uses irregular channel and spatial sizes to
    exercise lazy initialization and non-trivial tensor shapes.
    """
    def __init__(
        self,
        in_channels: int,
        conv_out: int,
        kernel_size: int,
        stride: int,
        padding: int,
        threshold_value: float,
        threshold_negval: float,
        linear_out: int,
    ):
        super(Model, self).__init__()
        # Primary convolution path
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=conv_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )

        # 1x1 convolution used as a residual/skip path to match channels
        self.short_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=conv_out,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        # SiLU activation (Swish)
        self.silu = nn.SiLU()

        # Lazy InstanceNorm2d: will infer num_features on first forward pass
        # Use affine=True so scale and bias are learnable after initialization
        self.inst_norm = nn.LazyInstanceNorm2d(eps=1e-5, affine=True)

        # Threshold activation to zero-out values below threshold_value (or set to threshold_negval)
        self.threshold = nn.Threshold(threshold_value, threshold_negval)

        # Final fully connected projection from channel dimension after global pooling
        self.fc = nn.Linear(conv_out, linear_out, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        1. Primary conv -> SiLU
        2. Shortcut 1x1 conv
        3. Add residual
        4. Lazy instance norm (initialized on first call)
        5. Threshold activation
        6. Global spatial average pooling (H,W) -> (B, C)
        7. Linear projection -> (B, linear_out)
        """
        # Primary branch
        y = self.conv1(x)         # Conv2d
        y = self.silu(y)          # SiLU activation

        # Shortcut branch to match channels and enable residual-style addition
        s = self.short_conv(x)    # 1x1 Conv2d

        # Residual combination
        y = y + s                 # Element-wise add

        # Lazy InstanceNorm2d initializes num_features based on y.shape[1] at first call
        y = self.inst_norm(y)     # Instance normalization (per-sample, per-channel)

        # Threshold to clip small activations (keeps values >= threshold_value, else set to threshold_negval)
        y = self.threshold(y)     # Threshold

        # Global spatial average pooling to collapse H and W
        y = y.mean(dim=[2, 3])    # Shape: (batch_size, conv_out)

        # Final linear layer
        out = self.fc(y)          # Shape: (batch_size, linear_out)
        return out

# Configuration / module-level variables (irregular sizes to stress lazy init)
batch_size = 8
in_channels = 3
height = 97
width = 65

conv_out = 83           # non-power-of-two, irregular channel size
kernel_size = 3
conv_stride = 1
conv_padding = 1

threshold_value = 0.05
threshold_negval = -1.0

linear_out = 128

def get_inputs():
    """
    Returns a list containing the primary input tensor for the model.
    Shape: (batch_size, in_channels, height, width)
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters for the Model constructor:
    [in_channels, conv_out, kernel_size, conv_stride, conv_padding, threshold_value, threshold_negval, linear_out]
    """
    return [in_channels, conv_out, kernel_size, conv_stride, conv_padding, threshold_value, threshold_negval, linear_out]