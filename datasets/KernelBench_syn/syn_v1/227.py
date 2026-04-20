import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex module that:
      - Upsamples a 3D volume using ConvTranspose3d
      - Collapses the depth dimension via mean to form a 2D feature map
      - Applies BatchNorm2d and a non-linearity
      - Applies RMSNorm across channels (treating channels as last dimension)
      - Projects features with a 1x1 Conv2d and pools to a compact vector

    Input:
      x: Tensor of shape (N, C_in, D, H, W)

    Output:
      Tensor of shape (N, out_channels) -- pooled per-example feature vector
    """
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int,
                 upscale: tuple = (2, 2, 2), kernel_size: tuple = (2, 2, 2)):
        super(Model, self).__init__()
        # Transposed convolution to upsample the 3D volume
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=upscale,
            padding=0,
            bias=True
        )
        # BatchNorm for the 2D feature map after collapsing depth
        self.bn2d = nn.BatchNorm2d(num_features=mid_channels)
        # RMSNorm to normalize across channels; will be applied after permuting channels to last dim
        self.rmsnorm = nn.RMSNorm(normalized_shape=mid_channels, eps=1e-8, elementwise_affine=True)
        # 1x1 Conv2d to project to desired output channels
        self.project_1x1 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, bias=True)
        # Non-linearity and final pooling
        self.gelu = nn.GELU()
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
          1. Upsample 3D volume with conv transpose -> (N, mid, D2, H2, W2)
          2. Collapse depth by mean -> (N, mid, H2, W2)
          3. BatchNorm2d -> GELU
          4. Permute to (..., mid) and apply RMSNorm (normalizes across channels)
          5. Permute back -> 1x1 conv projection -> global avg pool -> flatten
        """
        # 1) Upsample 3D volume
        x3d = self.conv_transpose3d(x)  # (N, mid_channels, D2, H2, W2)

        # 2) Collapse depth dimension by mean to produce a 2D feature map
        x2d = x3d.mean(dim=2)  # mean over depth -> (N, mid_channels, H2, W2)

        # 3) BatchNorm2d and non-linearity
        x_bn = self.bn2d(x2d)
        x_act = self.gelu(x_bn)

        # 4) RMSNorm across channels: move channels to last dimension
        #    Shape before: (N, mid, H, W) -> permute -> (N, H, W, mid)
        x_perm = x_act.permute(0, 2, 3, 1).contiguous()
        x_norm = self.rmsnorm(x_perm)  # normalizes last dim (=mid_channels)
        # permute back to (N, mid, H, W)
        x_back = x_norm.permute(0, 3, 1, 2).contiguous()

        # 5) 1x1 conv projection and global pooling to (N, out_channels)
        x_proj = self.project_1x1(x_back)  # (N, out_channels, H, W)
        x_pooled = self.pool(x_proj)       # (N, out_channels, 1, 1)
        out = x_pooled.view(x_pooled.size(0), -1)  # (N, out_channels)
        return out

# Configuration
batch_size = 8
in_channels = 3
depth = 4       # small depth that will be upsampled
height = 32
width = 32
mid_channels = 48
out_channels = 128

def get_inputs():
    """
    Returns a list containing:
      - A single 5D tensor appropriate for ConvTranspose3d input: (N, C_in, D, H, W)
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters for the Model constructor in the same order.
    """
    return [in_channels, mid_channels, out_channels]