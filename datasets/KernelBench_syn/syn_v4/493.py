import torch
import torch.nn as nn

# Configuration
batch_size = 4
in_channels = 3
depth = 8
height = 64
width = 64

class Model(nn.Module):
    """
    Complex model that combines 3D replication padding, 3D max pooling,
    a channel-depth reorganization, and a lazy 2D convolution followed by
    normalization and activation. The model demonstrates interaction
    between 3D and 2D modules by folding the depth dimension into
    the channel axis before applying a LazyConv2d.
    """
    def __init__(self, conv_out_channels: int = 16, conv_kernel: int = 3):
        super(Model, self).__init__()
        # Pad (W_left, W_right, H_top, H_bottom, D_front, D_back)
        self.pad3d = nn.ReplicationPad3d((1, 1, 2, 2, 1, 1))
        # Downsample spatially and along depth
        self.pool3d = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # LazyConv2d will determine in_channels on first forward pass.
        # We choose a small out_channels for efficiency; kernel_size with padding to preserve H/W dims.
        self.conv2d = nn.LazyConv2d(out_channels=conv_out_channels,
                                    kernel_size=conv_kernel,
                                    stride=1,
                                    padding=conv_kernel // 2,
                                    bias=False)
        # Normalize across conv output channels
        self.bn2d = nn.BatchNorm2d(conv_out_channels)
        # Activation
        self.act = nn.GELU()
        # Final adaptive pooling to produce a compact vector per sample
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        1. Apply replication padding in 3D.
        2. MaxPool3d to reduce depth/height/width.
        3. Permute and merge depth into channels to form a 4D tensor suitable for Conv2d.
        4. Apply LazyConv2d (in_channels inferred at first call), BatchNorm2d, and GELU.
        5. Global average pool to produce (batch, channels) output.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, D, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (N, conv_out_channels)
        """
        # Expecting shape (N, C, D, H, W)
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input (N, C, D, H, W), got shape {tuple(x.shape)}")

        # 1) Replication padding in 3D
        x_padded = self.pad3d(x)

        # 2) 3D MaxPool
        x_pooled = self.pool3d(x_padded)  # (N, C, D', H', W')

        # 3) Move depth into channel dimension:
        #    from (N, C, D', H', W') -> (N, D', C, H', W') -> (N, D'*C, H', W')
        n, c, d, h, w = x_pooled.shape
        x_perm = x_pooled.permute(0, 2, 1, 3, 4)
        x_4d = x_perm.reshape(n, d * c, h, w)

        # 4) 2D convolution, normalization, activation
        x_conv = self.conv2d(x_4d)  # LazyConv2d will set in_channels = d*c on first run
        x_norm = self.bn2d(x_conv)
        x_act = self.act(x_norm)

        # 5) Global pooling to (N, C_out, 1, 1) -> squeeze to (N, C_out)
        x_pooled2d = self.global_pool(x_act)
        out = x_pooled2d.view(n, -1)

        return out

def get_inputs():
    """
    Returns a list containing a single 5D input tensor with shape:
    (batch_size, in_channels, depth, height, width)
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    No special initialization inputs are required (LazyConv2d will infer in_channels at runtime).
    """
    return []