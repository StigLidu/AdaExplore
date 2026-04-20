import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex 3D->2D hybrid model that:
      - Applies MaxPool3d (with indices) over a 5D tensor (B, C, D, H, W)
      - Uses ReLU non-linearity on the pooled result
      - Uses MaxUnpool3d to restore the spatial/depth size
      - Merges the depth dimension into channels to form a 4D tensor (B, C*D, H, W)
      - Applies ConstantPad2d with a specified constant value
      - Runs a 2D convolution over the padded result

    This pipeline demonstrates interaction between 3D pooling/unpooling and 2D convolutional processing.
    """
    def __init__(
        self,
        in_channels: int,
        depth: int,
        out_channels: int,
        conv_kernel: int = 3,
        pool_kernel: int = 2,
        pad_value: float = 0.0,
    ):
        """
        Args:
            in_channels (int): Number of channels in the 3D input (C).
            depth (int): Depth dimension size of the input (D). Used to set Conv2d in_channels = C * D.
            out_channels (int): Number of output channels for the Conv2d.
            conv_kernel (int): Kernel size for the 2D convolution.
            pool_kernel (int): Kernel size / stride for 3D pooling/unpooling.
            pad_value (float): Constant value to use for ConstantPad2d.
        """
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.depth = depth
        self.pool_kernel = pool_kernel

        # 3D pooling and its corresponding unpool
        self.pool3d = nn.MaxPool3d(kernel_size=pool_kernel, stride=pool_kernel, return_indices=True)
        self.unpool3d = nn.MaxUnpool3d(kernel_size=pool_kernel, stride=pool_kernel)

        # Pad spatial dims (H, W) with a constant before Conv2d
        pad_amount = conv_kernel // 2  # preserve spatial size if desired
        self.const_pad2d = nn.ConstantPad2d(pad_amount, pad_value)

        # Conv2d merges the depth into channels: in_channels * depth -> out_channels
        self.conv2d = nn.Conv2d(in_channels * depth, out_channels, kernel_size=conv_kernel, bias=True)

        # Small normalization for stability
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, D, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (B, out_channels, H', W')
        """
        # MaxPool3d with indices
        pooled, indices = self.pool3d(x)  # pooled: (B, C, D//k, H//k, W//k)

        # Non-linearity
        activated = torch.relu(pooled)

        # Unpool back to original size using indices
        # Provide output_size to ensure exact restoration to x.shape
        unpooled = self.unpool3d(activated, indices, output_size=x.size())  # (B, C, D, H, W)

        # Merge depth into channels to prepare for 2D conv: (B, C*D, H, W)
        B, C, D, H, W = unpooled.shape
        merged = unpooled.contiguous().view(B, C * D, H, W)

        # Constant padding over spatial dims
        padded = self.const_pad2d(merged)  # (B, C*D, H+2p, W+2p)

        # 2D convolution
        conv_out = self.conv2d(padded)  # (B, out_channels, H_out, W_out)

        # Channel-wise normalization and a final non-linearity
        # LayerNorm expects (N, C, H, W) -> normalized over last three dims per channel
        # Permute to (N, H, W, C) -> apply LayerNorm over last dim -> permute back
        out_perm = conv_out.permute(0, 2, 3, 1)
        out_norm = self.norm(out_perm).permute(0, 3, 1, 2)

        return torch.relu(out_norm)


# Module-level configuration variables (sizes chosen to be compatible with pooling/unpooling)
BATCH = 4
CHANNELS = 3
DEPTH = 4        # must be divisible by pool_kernel if we want exact division, but MaxUnpool3d with output_size handles restoration
HEIGHT = 32
WIDTH = 32

OUT_CHANNELS = 16
CONV_KERNEL = 3
POOL_KERNEL = 2
PAD_VALUE = 0.05

def get_inputs():
    """
    Returns the input 5D tensor required by the Model:
      - x shape: (BATCH, CHANNELS, DEPTH, HEIGHT, WIDTH)
    """
    x = torch.randn(BATCH, CHANNELS, DEPTH, HEIGHT, WIDTH)
    return [x]

def get_init_inputs():
    """
    Returns a list of initialization parameters for the Model constructor in the order:
      [in_channels, depth, out_channels, conv_kernel, pool_kernel, pad_value]
    """
    return [CHANNELS, DEPTH, OUT_CHANNELS, CONV_KERNEL, POOL_KERNEL, PAD_VALUE]