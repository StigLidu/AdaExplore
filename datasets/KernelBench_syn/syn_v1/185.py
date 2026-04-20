import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex 3D-to-5D processing module that:
    - Pads a 5D input (N, C, D, H, W) with replication padding (ReplicationPad3d)
    - Collapses the depth dimension into the channel dimension to produce a 4D tensor
    - Applies an Lp pooling (LPPool2d) over spatial HxW
    - Projects channels with a 1x1 convolution
    - Adds a depth-averaged skip connection projected to the same channel space
    - Applies a Hardsigmoid activation and reshapes back to a 5D tensor
    """
    def __init__(
        self,
        in_channels: int,
        depth: int,
        pad: int,
        lp_norm: int,
        pool_kernel: int,
        out_channels: int,
        out_depth: int
    ):
        """
        Args:
            in_channels (int): Number of channels C in input tensor.
            depth (int): Depth D in input tensor.
            pad (int): Padding to apply on each spatial and depth side (symmetric).
            lp_norm (int): The p value for LPPool2d.
            pool_kernel (int): Kernel size (and stride) for LPPool2d.
            out_channels (int): Number of output channels after 1x1 conv.
            out_depth (int): Desired output depth to split channels into (must divide out_channels).
        """
        super(Model, self).__init__()
        assert out_channels % out_depth == 0, "out_channels must be divisible by out_depth"
        # ReplicationPad3d expects (left, right, top, bottom, front, back)
        self.pad = nn.ReplicationPad3d((pad, pad, pad, pad, pad, pad))
        self.lp_pool = nn.LPPool2d(norm_type=lp_norm, kernel_size=pool_kernel)
        # After padding depth becomes depth + 2*pad; we collapse depth into channels in forward, so conv in_channels reflects that
        collapsed_in_ch = in_channels * (depth + 2 * pad)
        self.conv1 = nn.Conv2d(collapsed_in_ch, out_channels, kernel_size=1)
        # Skip path projects depth-averaged original to same out_channels for residual addition
        self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.hsigmoid = nn.Hardsigmoid()
        self.pool_kernel = pool_kernel
        self.pad_amount = pad
        self.in_channels = in_channels
        self.depth = depth
        self.out_channels = out_channels
        self.out_depth = out_depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input of shape (N, C, D, H, W)

        Returns:
            torch.Tensor: Output of shape (N, out_channels/out_depth, out_depth, H_out, W_out)
        """
        # x: (N, C, D, H, W)
        N, C, D, H, W = x.shape

        # 1) Replicate-pad the 3D spatial dimensions (depth, height, width)
        x_padded = self.pad(x)  # (N, C, D + 2*pad, H + 2*pad, W + 2*pad)
        _, _, Dp, Hp, Wp = x_padded.shape

        # 2) Collapse depth into channels to produce a 4D tensor for 2D pooling
        collapsed = x_padded.reshape(N, C * Dp, Hp, Wp)  # (N, C*Dp, Hp, Wp)

        # 3) Apply Lp pooling over spatial dimensions (reduces Hp/Wp according to pool_kernel)
        pooled = self.lp_pool(collapsed)  # (N, C*Dp, Hp//k, Wp//k)
        Hp2 = pooled.shape[2]
        Wp2 = pooled.shape[3]

        # 4) 1x1 convolution to mix collapsed channels into out_channels
        projected = self.conv1(pooled)  # (N, out_channels, Hp2, Wp2)

        # 5) Skip connection: average original input over depth, adaptively pool to same spatial dims, project
        skip = x.mean(dim=2)  # (N, C, H, W) -- depth-averaged
        # adapt to padded spatial resolution divided by pool_kernel
        # target spatial size is (Hp2, Wp2)
        skip_pooled = F.adaptive_avg_pool2d(F.pad(skip, (self.pad_amount, self.pad_amount, self.pad_amount, self.pad_amount), mode='replicate'), (Hp2, Wp2))
        skip_proj = self.skip_conv(skip_pooled)  # (N, out_channels, Hp2, Wp2)

        # 6) Residual add and non-linearity
        combined = projected + skip_proj
        activated = self.hsigmoid(combined)  # elementwise bounded activation

        # 7) Reshape back to 5D by splitting channels into (channels_per_slice, depth_slices, H_out, W_out)
        channels_per_slice = self.out_channels // self.out_depth
        out = activated.reshape(N, channels_per_slice, self.out_depth, Hp2, Wp2)
        return out

# Configuration variables
batch_size = 8
in_channels = 4
depth = 3
height = 32
width = 32
pad = 1
lp_norm = 2      # p-value for Lp pooling
pool_kernel = 2  # kernel size for Lp pooling (and stride)
out_channels = 16
out_depth = 4    # must divide out_channels

def get_inputs():
    # Create a random 5D tensor (N, C, D, H, W)
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    # Initialization parameters in the same order as Model.__init__
    return [in_channels, depth, pad, lp_norm, pool_kernel, out_channels, out_depth]