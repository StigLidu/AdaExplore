import torch
import torch.nn as nn

# Configuration / module-level variables
batch_size = 8
in_channels = 3
depth = 16
height = 32
width = 32

# Adaptive pooling target (must be compatible with PixelUnshuffle downscale)
pool_d = 8
pool_h = 16  # should be divisible by UNSHUFFLE_FACTOR
pool_w = 16  # should be divisible by UNSHUFFLE_FACTOR

UNSHUFFLE_FACTOR = 2  # PixelUnshuffle downscale factor
CONV_OUT_CHANNELS = 64  # out channels for LazyConv3d

class Model(nn.Module):
    """
    Complex 3D processing model that:
      1) Applies an AdaptiveMaxPool3d to reduce spatial dims.
      2) Uses PixelUnshuffle on each depth slice (batched via reshape) to increase channels and downscale XY.
      3) Processes the result with a LazyConv3d (lazy in_channels).
      4) Applies GELU activation and final AdaptiveMaxPool3d to produce a compact embedding per batch.
    Input:
        x: Tensor of shape (N, C, D, H, W)
    Output:
        Tensor of shape (N, CONV_OUT_CHANNELS) -- global descriptors per batch element
    """
    def __init__(self):
        super(Model, self).__init__()
        # First spatial reduction to (pool_d, pool_h, pool_w)
        self.pool1 = nn.AdaptiveMaxPool3d((pool_d, pool_h, pool_w))
        # PixelUnshuffle used on 4D tensors (N*D, C, H, W); downscales H/W and increases channels by r^2
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=UNSHUFFLE_FACTOR)
        # Lazy Conv3d will infer in_channels on first forward from input
        self.conv3d = nn.LazyConv3d(out_channels=CONV_OUT_CHANNELS, kernel_size=3, padding=1)
        # Activation and final spatial pooling to (1,1,1) -> reduce to per-channel descriptors
        self.act = nn.GELU()
        self.pool2 = nn.AdaptiveMaxPool3d((1, 1, 1))
        # A small learnable scale applied to final descriptors
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining pooling, pixel unshuffle, 3D conv, activation, and pooling.

        Args:
            x: Input tensor (N, C, D, H, W)

        Returns:
            Tensor of shape (N, CONV_OUT_CHANNELS)
        """
        # Step 1: reduce spatial dims
        x = self.pool1(x)  # (N, C, pool_d, pool_h, pool_w)

        N, C, Dp, Hp, Wp = x.shape
        assert Hp % UNSHUFFLE_FACTOR == 0 and Wp % UNSHUFFLE_FACTOR == 0, \
            "Hp and Wp must be divisible by UNSHUFFLE_FACTOR"

        # Step 2: prepare for PixelUnshuffle by merging batch and depth
        # Move depth next to batch: (N, Dp, C, Hp, Wp) -> reshape to (N*Dp, C, Hp, Wp)
        x_resh = x.permute(0, 2, 1, 3, 4).contiguous().view(N * Dp, C, Hp, Wp)

        # Apply PixelUnshuffle: channels become C * r^2, spatial dims shrink by r
        x_unsh = self.pixel_unshuffle(x_resh)  # (N*Dp, C*r^2, Hp/r, Wp/r)

        # Step 3: restore depth dimension: (N, Dp, C*r^2, Hp/r, Wp/r)
        new_C = C * (UNSHUFFLE_FACTOR ** 2)
        Hp_r = Hp // UNSHUFFLE_FACTOR
        Wp_r = Wp // UNSHUFFLE_FACTOR
        x_unsh = x_unsh.view(N, Dp, new_C, Hp_r, Wp_r).permute(0, 2, 1, 3, 4).contiguous()
        # Now x_unsh shape: (N, new_C, Dp, Hp_r, Wp_r) suitable for Conv3d

        # Step 4: 3D convolution (LazyConv3d will initialize in_channels = new_C on first forward)
        x_conv = self.conv3d(x_unsh)  # (N, CONV_OUT_CHANNELS, Dp, Hp_r, Wp_r)

        # Step 5: activation
        x_act = self.act(x_conv)

        # Step 6: global spatial pooling to (1,1,1) and squeeze to (N, CONV_OUT_CHANNELS)
        x_pooled = self.pool2(x_act).view(N, CONV_OUT_CHANNELS)

        # Final scaled descriptor
        out = x_pooled * self.scale

        return out

def get_inputs():
    """
    Returns the example input tensor(s) for the model.
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Returns any initialization parameters required externally.
    (None required here since LazyConv3d initializes itself lazily.)
    """
    return []