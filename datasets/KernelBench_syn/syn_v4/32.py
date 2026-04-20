import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex convolutional block that demonstrates:
    - Two-stage convolutional feature extraction
    - Channel-wise dropout (Dropout2d) for regularization
    - Spatial Lp-pooling (LPPool2d) to reduce spatial dimensions with tunable p-norm
    - A learnable shortcut (1x1 conv) to match channels and downsample the residual
    The block returns a residual-summed output with a final non-linearity.
    """
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        conv_kernel: int = 3,
        pool_kernel: int = 2,
        lp_norm: float = 2.0,
        dropout_p: float = 0.2,
    ):
        """
        Initializes the module.

        Args:
            in_channels: Number of input channels.
            mid_channels: Number of channels in the intermediate conv.
            out_channels: Number of output channels.
            conv_kernel: Kernel size for the 3x3 convolutions (assumed odd).
            pool_kernel: Kernel / stride for LPPool2d (spatial downsampling factor).
            lp_norm: p value for LPPool2d (e.g., 1, 2, inf).
            dropout_p: Dropout probability for Dropout2d.
        """
        super(Model, self).__init__()

        padding = (conv_kernel - 1) // 2

        # First convolution: keep spatial resolution
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=conv_kernel, padding=padding, bias=True)
        # Second convolution: produce desired output channels
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=conv_kernel, padding=padding, bias=True)

        # Shortcut to match channels and downsample spatially to align with LPPool2d output
        # Use 1x1 conv with stride equal to pool_kernel to reduce spatial dims
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=pool_kernel, bias=True)

        # Regularization and pooling
        self.dropout2d = nn.Dropout2d(p=dropout_p)
        self.lppool = nn.LPPool2d(norm_type=lp_norm, kernel_size=pool_kernel, stride=pool_kernel)

        # Small epsilon to avoid division by zero if we later normalize
        self.eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the composed block.

        Pattern:
            1) conv1 -> ReLU
            2) conv2 -> ReLU
            3) dropout2d
            4) LPPool2d (spatial downsampling)
            5) shortcut downsample of original input
            6) elementwise addition (residual) -> final ReLU
            7) channel-wise L2 normalization (optional lightweight normalization)

        Args:
            x: Input tensor of shape (N, C_in, H, W).

        Returns:
            Tensor of shape (N, C_out, H / pool_kernel, W / pool_kernel)
        """
        # Main path
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)

        # Regularize channels
        out = self.dropout2d(out)

        # Spatial pooling (reduces H/W by pool_kernel)
        out = self.lppool(out)

        # Residual / shortcut path: downsample and match channels
        res = self.shortcut(x)

        # Combine
        out = out + res
        out = F.relu(out)

        # Channel-wise L2 normalization: divide each channel by its L2 norm across spatial dims
        # Compute norms per-sample and per-channel: shape (N, C, 1, 1)
        # add eps for numerical stability
        norm = torch.sqrt(torch.sum(out * out, dim=(2, 3), keepdim=True) + self.eps)
        out = out / norm

        return out

# Top-level configuration variables (example sizes)
batch_size = 8
in_channels = 3
mid_channels = 32
out_channels = 64
height = 128
width = 128
conv_kernel = 3
pool_kernel = 2
lp_norm = 2.0
dropout_p = 0.25

def get_inputs():
    """
    Returns example input tensors for running the model.
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters to pass to the Model constructor.
    The order matches the __init__ signature:
      in_channels, mid_channels, out_channels, conv_kernel, pool_kernel, lp_norm, dropout_p
    """
    return [in_channels, mid_channels, out_channels, conv_kernel, pool_kernel, lp_norm, dropout_p]