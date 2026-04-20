import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex example combining a lazy 3D convolution with 2D pooling/unpooling and nearest-neighbor upsampling.
    The model:
      - Applies a LazyConv3d to the input (B, C_in, D, H, W) producing (B, C_out, D, H, W)
      - Collapses the channel dimension to produce a single-channel per-depth 2D map
      - Performs MaxPool2d with indices, then MaxUnpool2d to recover spatial locations
      - Applies nearest-neighbor upsampling on the unpooled maps to expand spatial resolution
      - Uses the upsampled map as an attention-like modulation on a nearest-neighbor upsampled conv feature map
    Returns a tensor of shape (B, C_out, D, H * up_scale, W * up_scale).
    """
    def __init__(self,
                 out_channels: int,
                 kernel_size3d=(3, 3, 3),
                 pool_kernel: int = 2,
                 up_scale: int = 2):
        """
        Args:
            out_channels (int): Number of output channels for the LazyConv3d.
            kernel_size3d (tuple): 3D convolution kernel size.
            pool_kernel (int): Kernel (and stride) size for 2D max pooling/unpooling.
            up_scale (int): Nearest-neighbor upsampling scale factor for the attention maps and conv features.
        """
        super(Model, self).__init__()
        # LazyConv3d will infer in_channels on first forward pass
        padding = tuple(k // 2 for k in kernel_size3d)
        self.conv3d = nn.LazyConv3d(out_channels, kernel_size=kernel_size3d, padding=padding)
        self.pool2d = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_kernel, return_indices=True)
        self.unpool2d = nn.MaxUnpool2d(kernel_size=pool_kernel, stride=pool_kernel)
        self.upsample2d = nn.UpsamplingNearest2d(scale_factor=up_scale)

        # Keep parameters for use in forward (useful for interpolation scaling)
        self.out_channels = out_channels
        self.kernel_size3d = kernel_size3d
        self.pool_kernel = pool_kernel
        self.up_scale = up_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, D, H * up_scale, W * up_scale).
        """
        B, _, D, H, W = x.shape  # input channel inferred by LazyConv3d on first call

        # 1) 3D convolution to produce feature map
        conv_out = self.conv3d(x)  # shape: (B, C_out, D, H, W)

        # 2) Create a single-channel 2D map per depth by averaging across channels
        #    Shape -> (B, 1, D, H, W)
        mean_map = conv_out.mean(dim=1, keepdim=True)

        # 3) Reshape to 2D batch (combine batch and depth) for 2D pooling
        #    Shape -> (B * D, 1, H, W)
        mean_map_2d = mean_map.permute(0, 2, 1, 3, 4).reshape(B * D, 1, H, W)

        # 4) MaxPool2d with indices -> pooled (B*D, 1, Hp, Wp), indices for unpooling
        pooled, indices = self.pool2d(mean_map_2d)

        # 5) MaxUnpool2d to restore to original 2D spatial size (H, W)
        #    output_size provided as full tensor shape works reliably
        unpooled = self.unpool2d(pooled, indices, output_size=mean_map_2d.shape)

        # 6) Upsample the unpooled map with nearest-neighbor to create an attention map at higher resolution
        upsampled_att = self.upsample2d(unpooled)  # shape: (B*D, 1, H_up, W_up)
        H_up, W_up = upsampled_att.size(2), upsampled_att.size(3)

        # 7) Reshape attention back to (B, 1, D, H_up, W_up)
        up_att = upsampled_att.view(B, D, 1, H_up, W_up).permute(0, 2, 1, 3, 4)

        # 8) Create an attention mask in (0,1)
        att_mask = torch.sigmoid(up_att)

        # 9) Nearest-neighbor upsample conv features across (H,W) while keeping D intact
        #    Uses F.interpolate which supports 5D tensors; scale_factor=(1, up_scale, up_scale)
        conv_up = F.interpolate(conv_out, scale_factor=(1, self.up_scale, self.up_scale), mode='nearest')

        # 10) Modulate conv features with attention and apply non-linearity
        out = conv_up * (1.0 + att_mask)  # broadcasting over the channel dimension
        out = F.relu(out)

        return out

# Configuration for generating inputs and initialization
batch_size = 4
in_channels = 3
depth = 8
height = 32
width = 32

out_channels = 16
kernel3d = (3, 3, 3)
pool_k = 2
up_scale = 2

def get_inputs():
    """
    Returns example input tensors for a forward pass.
    x shape: (batch_size, in_channels, depth, height, width)
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for the Model constructor:
      [out_channels, kernel_size3d, pool_kernel, up_scale]
    """
    return [out_channels, kernel3d, pool_k, up_scale]