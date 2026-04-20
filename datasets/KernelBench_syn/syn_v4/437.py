import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex 2D feature processor that combines MaxPool2d, Softplus, and ZeroPad3d
    with a learned channel mixing and a residual connection.

    Computation pipeline:
      1. MaxPool2d over spatial dims (H, W)
      2. Softplus non-linearity
      3. Insert a singleton depth dimension and apply ZeroPad3d (depth, height, width)
      4. Mean over the depth dimension to collapse back to 4D
      5. Channel mixing via a learnable linear transform (performs channel projection)
      6. Upsample spatial dimensions back to the original input size
      7. Residual addition with the original input and final Softplus
    """
    def __init__(self, in_channels: int, pool_kernel: tuple, pool_stride: tuple, pad3d: tuple, out_channels: int = None):
        """
        Args:
            in_channels (int): Number of input channels (and default output channels).
            pool_kernel (tuple): Kernel size for MaxPool2d (kh, kw).
            pool_stride (tuple): Stride for MaxPool2d (sh, sw).
            pad3d (tuple): 6-tuple for ZeroPad3d (pad_dL, pad_dR, pad_hL, pad_hR, pad_wL, pad_wR).
            out_channels (int, optional): Number of output channels after mixing. Defaults to in_channels.
        """
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels

        # Spatial pooling
        self.maxpool = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride)

        # Non-linearity
        self.softplus = nn.Softplus()

        # 3D zero padding (we insert a depth dim of size 1 before using this)
        self.pad3d = nn.ZeroPad3d(pad3d)

        # Learnable channel mixing: projects from in_channels -> out_channels
        # Shape: (in_channels, out_channels) for einsum convenience
        self.channel_weight = nn.Parameter(torch.randn(in_channels, self.out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (N, out_channels, H, W)
        """
        N, C, H, W = x.shape
        # 1) Spatial max pooling
        p = self.maxpool(x)  # (N, C, Hp, Wp)

        # 2) Softplus activation
        s = self.softplus(p)  # (N, C, Hp, Wp)

        # 3) Insert depth dim and apply ZeroPad3d
        s5 = s.unsqueeze(2)  # (N, C, D=1, Hp, Wp)
        z = self.pad3d(s5)   # (N, C, D', Hp', Wp')

        # 4) Collapse depth dim by averaging
        dmean = z.mean(dim=2)  # (N, C, Hp', Wp')

        # 5) Channel mixing via learned linear projection
        # Use einsum to multiply across channel dimension: (N, C, H, W) x (C, F) -> (N, F, H, W)
        mixed = torch.einsum('nchw,cf->nfhw', dmean, self.channel_weight)

        # 6) Upsample spatial dims back to original H, W
        up = F.interpolate(mixed, size=(H, W), mode='bilinear', align_corners=False)  # (N, F, H, W)

        # 7) If channel counts match, add residual from input; otherwise, broadcast-add first channels
        if self.out_channels == C:
            res = up + x
        else:
            # If out_channels != in_channels, pad or truncate channels from input to match out_channels
            if self.out_channels < C:
                res = up + x[:, :self.out_channels, :, :]
            else:
                # self.out_channels > C: pad input channels with zeros
                pad_shape = (N, self.out_channels - C, H, W)
                zero_pad = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
                x_padded = torch.cat([x, zero_pad], dim=1)
                res = up + x_padded

        # Final non-linearity
        return self.softplus(res)


# Module-level configuration
BATCH_SIZE = 4
IN_CHANNELS = 16
HEIGHT = 64
WIDTH = 64

POOL_KERNEL = (3, 3)
POOL_STRIDE = (2, 2)

# ZeroPad3d expects (pad_dL, pad_dR, pad_hL, pad_hR, pad_wL, pad_wR)
# We pad depth by 1 on each side (starting from depth=1), and pad spatial dims slightly
PAD3D = (1, 1, 1, 1, 1, 1)

def get_inputs():
    """
    Returns a list with the primary input tensor to the model.
    """
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for the Model constructor in the order:
    (in_channels, pool_kernel, pool_stride, pad3d)

    These can be used to instantiate the model as:
        model = Model(*get_init_inputs())
    """
    return [IN_CHANNELS, POOL_KERNEL, POOL_STRIDE, PAD3D]