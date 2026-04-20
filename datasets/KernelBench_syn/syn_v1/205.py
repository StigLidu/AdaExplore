import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Composite model that demonstrates a mix of 2D lazy convolution, fractional max pooling,
    and 3D adaptive max pooling. It fuses spatial summaries from a 2D branch and a 3D branch
    into a compact feature vector and projects it back to a channel-sized output.
    """
    def __init__(
        self,
        out_channels: int,
        c3: int,
        adaptive_output_size: tuple,
        frac_output_ratio: tuple,
        conv_kernel: int = 3,
        frac_kernel: tuple = (2, 2),
    ):
        """
        Args:
            out_channels (int): Number of output channels produced by the 2D convolution.
                                Also used as the final projected feature size.
            c3 (int): Number of channels in the 3D input branch.
            adaptive_output_size (tuple): Output size for AdaptiveMaxPool3d (D_out, H_out, W_out).
            frac_output_ratio (tuple): Output ratio for FractionalMaxPool2d (h_ratio, w_ratio).
            conv_kernel (int): Kernel size for the LazyConv2d layer (default 3).
            frac_kernel (tuple): Kernel size for FractionalMaxPool2d (default (2,2)).
        """
        super(Model, self).__init__()

        # 2D branch: lazy conv so in_channels can be determined at first forward pass
        self.conv2d = nn.LazyConv2d(out_channels=out_channels, kernel_size=conv_kernel, padding=conv_kernel // 2)
        # Non-linearity
        self.relu = nn.ReLU()
        # Fractional pooling reduces 2D spatial dims by approximate ratio
        self.frac_pool2d = nn.FractionalMaxPool2d(kernel_size=frac_kernel, output_ratio=frac_output_ratio, return_indices=False)

        # 3D branch: adaptive pooling to a fixed small 3D grid
        self.adapt3d = nn.AdaptiveMaxPool3d(adaptive_output_size)

        # Fusion linear projection: (out_channels from 2D summary) + (c3 from 3D summary) -> out_channels
        fused_dim = out_channels + c3
        self.fc = nn.Linear(fused_dim, out_channels)

    def forward(self, x2d: torch.Tensor, x3d: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that:
        - Applies a LazyConv2d -> ReLU on the 2D input
        - Applies FractionalMaxPool2d to the conv output and reduces spatial dims by max
        - Applies AdaptiveMaxPool3d to the 3D input and reduces to channel summaries by average
        - Concatenates the channel summaries and projects them back to out_channels

        Args:
            x2d: Tensor of shape (N, C_in, H, W)
            x3d: Tensor of shape (N, C3, D, H3, W3)

        Returns:
            Tensor of shape (N, out_channels)
        """
        # 2D branch
        y = self.conv2d(x2d)           # (N, out_channels, H, W)  (LazyConv2d initializes here if needed)
        y = self.relu(y)
        y = self.frac_pool2d(y)        # (N, out_channels, H', W')
        # Channel-wise summary via max over spatial dims -> (N, out_channels)
        y_vec = torch.amax(y, dim=(2, 3))

        # 3D branch
        z = self.adapt3d(x3d)          # (N, C3, D_out, H_out, W_out)
        # Channel-wise summary via mean over 3D spatial dims -> (N, C3)
        z_vec = torch.mean(z, dim=(2, 3, 4))

        # Fuse and project
        fused = torch.cat([y_vec, z_vec], dim=1)  # (N, out_channels + C3)
        out = self.fc(fused)                      # (N, out_channels)
        out = self.relu(out)
        return out

# Configuration / default sizes
batch_size = 8
in_channels = 3     # for the 2D input (will be lazily registered in the conv)
out_channels = 64   # target channel size produced by the conv and final projection

c3 = 16             # channels for the 3D input
D = 8
H = 128
W = 128

H3 = 16
W3 = 16

adaptive_output_size = (4, 4, 4)   # D_out, H_out, W_out for AdaptiveMaxPool3d
frac_output_ratio = (0.5, 0.5)     # reduce H and W roughly by half in fractional pooling

def get_inputs():
    """
    Returns example input tensors:
    - x2d: (batch_size, in_channels, H, W)
    - x3d: (batch_size, c3, D, H3, W3)
    """
    x2d = torch.randn(batch_size, in_channels, H, W)
    x3d = torch.randn(batch_size, c3, D, H3, W3)
    return [x2d, x3d]

def get_init_inputs():
    """
    Returns the initialization arguments for the Model constructor:
    [out_channels, c3, adaptive_output_size, frac_output_ratio]
    """
    return [out_channels, c3, adaptive_output_size, frac_output_ratio]