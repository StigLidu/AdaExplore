import torch
import torch.nn as nn

# Module-level configuration
batch_size = 8
in_channels = 3
mid_channels = 32
out_channels = 16
height = 64
width = 64
conv_kernel = 3
pool_p = 2
lp_kernel = 2
depth_repeat = 4  # how many slices to replicate into the depth dimension before 3D pooling

class Model(nn.Module):
    """
    Complex model that combines 2D convolution, 3D Lp pooling and lazy InstanceNorm1d.
    Computation pattern:
      - 2D convolution + ReLU
      - expand to 5D by repeating along a new depth dimension
      - LPPool3d to reduce depth+spatial resolution
      - collapse spatial dims and apply LazyInstanceNorm1d over channel dimension
      - restore spatial dims, average over depth and finish with a 1x1 Conv2d
    """
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        conv_kernel: int = 3,
        pool_p: int = 2,
        lp_kernel: int = 2,
        depth_repeat: int = 4,
    ):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.conv_kernel = conv_kernel
        self.pool_p = pool_p
        self.lp_kernel = lp_kernel
        self.depth_repeat = depth_repeat

        # First 2D convolution to extract spatial features
        pad = conv_kernel // 2
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=conv_kernel, padding=pad, bias=True)

        # Lp pooling in 3D (operates on N, C, D, H, W)
        # Using stride == kernel_size for downsampling behavior
        self.lp_pool3d = nn.LPPool3d(norm_type=pool_p, kernel_size=lp_kernel, stride=lp_kernel)

        # LazyInstanceNorm1d will infer num_features during the first forward pass
        # It operates over (N, C, L) inputs, normalizing per-channel across length
        self.inst_norm1d = nn.LazyInstanceNorm1d()

        # Final 1x1 conv to mix channels after pooling/normalization
        self.conv_out = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (N, C_in, H, W)
        Returns:
            Tensor of shape (N, out_channels, H_out, W_out) after the described processing
        """
        # 1) 2D convolution + non-linearity
        x = self.conv1(x)                  # (N, mid_channels, H, W)
        x = torch.relu(x)

        # 2) Expand to 5D by inserting a depth dimension and repeating slices
        # This creates a pseudo-volume from a single 2D feature map
        x5 = x.unsqueeze(2).repeat(1, 1, self.depth_repeat, 1, 1)  # (N, C, D, H, W)

        # 3) Lp pooling over the 3D volume to reduce D, H, W
        p = self.lp_pool3d(x5)             # (N, C, D2, H2, W2)

        # 4) Collapse spatial dims into a single length dimension for 1D instance norm
        N, C, D2, H2, W2 = p.shape
        flat = p.view(N, C, -1)            # (N, C, L) where L = D2*H2*W2

        # 5) Apply LazyInstanceNorm1d which will initialize num_features lazily
        normed = self.inst_norm1d(flat)    # (N, C, L)

        # 6) Restore the 5D shape
        restored = normed.view(N, C, D2, H2, W2)  # (N, C, D2, H2, W2)

        # 7) Aggregate over the depth dimension to get back to 2D feature maps
        fused = restored.mean(dim=2)       # (N, C, H2, W2)

        # 8) Final 1x1 convolution to produce desired output channels
        out = self.conv_out(fused)         # (N, out_channels, H2, W2)

        return out

def get_inputs():
    """
    Returns a list containing the input tensor for the forward pass.
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for the Model constructor in order.
    """
    return [in_channels, mid_channels, out_channels, conv_kernel, pool_p, lp_kernel, depth_repeat]