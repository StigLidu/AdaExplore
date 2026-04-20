import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex module that upsamples with a transposed convolution,
    applies Hardsigmoid non-linearity and a Threshold, then performs
    a learned channel-wise re-scaling and a residual-style skip connection.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        threshold: float = 0.1,
    ):
        """
        Initializes the module components.

        Args:
            in_channels (int): Number of channels in the input.
            out_channels (int): Number of channels produced by the transposed conv.
            kernel_size (int): Kernel size for ConvTranspose2d (default 4 for 2x upsample).
            stride (int): Stride for ConvTranspose2d (default 2 for doubling spatial dims).
            padding (int): Padding for ConvTranspose2d.
            threshold (float): Threshold value for nn.Threshold.
        """
        super(Model, self).__init__()
        # Transposed convolution to upsample spatial dimensions
        self.deconv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )
        # Small 1x1 conv for skip connection channel matching
        self.skip_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        # Non-linearities and thresholding
        self.hardsigmoid = nn.Hardsigmoid()
        self.threshold = nn.Threshold(threshold, 0.0)

        # Learnable per-channel scale (applied after global pooling)
        self.channel_scale = nn.Parameter(torch.ones(1, out_channels, 1, 1) * 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        1. Upsample via ConvTranspose2d (deconv).
        2. Apply Hardsigmoid activation.
        3. Threshold small activations to zero.
        4. Compute per-channel global context, pass through Hardsigmoid scaled by a learnable param.
        5. Re-scale features by this context.
        6. Add a skip connection (input upsampled and 1x1 convolved).
        7. Final Hardsigmoid to keep outputs bounded.

        Args:
            x (torch.Tensor): Input tensor of shape (B, in_channels, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (B, out_channels, H*stride, W*stride)
        """
        # 1. Upsample with learned transposed convolution
        y = self.deconv(x)  # (B, out_channels, H*2, W*2)

        # 2. Non-linearity
        y = self.hardsigmoid(y)

        # 3. Threshold small values to exact zeros
        y = self.threshold(y)

        # 4. Global context: channel-wise mean over spatial dims
        context = y.mean(dim=(2, 3), keepdim=True)  # (B, out_channels, 1, 1)

        # 5. Apply learnable scaling and a bounded non-linearity to produce gating weights
        gates = self.hardsigmoid(context * self.channel_scale)

        # 6. Re-scale feature maps by gates (broadcast over spatial dims)
        y = y * gates

        # 7. Build skip connection: upsample input and match channels with 1x1 conv
        skip = F.interpolate(x, scale_factor=self.deconv.stride, mode='bilinear', align_corners=False)
        skip = self.skip_conv(skip)

        out = y + skip

        # 8. Final bounding activation to stabilize outputs
        out = self.hardsigmoid(out)

        return out

# Module-level configuration variables
batch_size = 8
in_channels = 32
out_channels = 64
height = 16
width = 16
kernel_size = 4
stride = 2
padding = 1
threshold_value = 0.1

def get_inputs():
    """
    Creates and returns a list with a single input tensor compatible with Model.forward.
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters for the Model constructor:
    (in_channels, out_channels, kernel_size, stride, padding, threshold)
    """
    return [in_channels, out_channels, kernel_size, stride, padding, threshold_value]