import torch
import torch.nn as nn

# Configuration
BATCH_SIZE = 2
IN_CHANNELS = 3
HEIGHT = 64
WIDTH = 64

PIXEL_UNSHUFFLE_R = 2          # downscale factor for PixelUnshuffle
DEPTH_SPLITS = 3               # number of slices to form the 3D depth dimension
DECONV_OUT_CHANNELS = 7        # out_channels for LazyConvTranspose3d
DECONV_KERNEL = (2, 3, 3)      # kernel size for ConvTranspose3d (D, H, W)
DECONV_STRIDE = (1, 2, 2)      # stride for ConvTranspose3d (D, H, W)
DECONV_PADDING = (0, 1, 1)     # padding for ConvTranspose3d (D, H, W)

POOL_KERNEL = (1, 2, 2)        # pooling only in spatial dims
POOL_STRIDE = (1, 2, 2)


class Model(nn.Module):
    """
    Complex model combining PixelUnshuffle, a LazyConvTranspose3d (deconvolution in 3D),
    MaxPool3d (with indices), and MaxUnpool3d to reconstruct spatial structure.
    The pipeline:
      1. PixelUnshuffle reduces H/W and increases channels.
      2. Channels are reinterpreted as a depth dimension to create a 5D tensor.
      3. A LazyConvTranspose3d upsamples (mostly spatially) and changes channel count.
      4. MaxPool3d is applied (returning indices), then immediately inverted with MaxUnpool3d.
      5. The depth dimension is folded back into channels producing a 4D tensor which is L2-normalized.
    """
    def __init__(self,
                 pixel_unshuffle_r: int = PIXEL_UNSHUFFLE_R,
                 depth_splits: int = DEPTH_SPLITS,
                 deconv_out_channels: int = DECONV_OUT_CHANNELS,
                 deconv_kernel=DECONV_KERNEL,
                 deconv_stride=DECONV_STRIDE,
                 deconv_padding=DECONV_PADDING,
                 pool_kernel=POOL_KERNEL,
                 pool_stride=POOL_STRIDE):
        super(Model, self).__init__()
        self.r = pixel_unshuffle_r
        self.depth_splits = depth_splits

        # PixelUnshuffle: reduces spatial resolution by factor r, increases channels by r^2
        self.pixel_unshuffle = nn.PixelUnshuffle(self.r)

        # LazyConvTranspose3d: in_channels will be inferred at first forward pass
        # We deliberately choose a kernel/stride to upsample the spatial dimensions.
        self.deconv3d = nn.LazyConvTranspose3d(out_channels=deconv_out_channels,
                                               kernel_size=deconv_kernel,
                                               stride=deconv_stride,
                                               padding=deconv_padding)

        # MaxPool3d -> MaxUnpool3d pair (we use return_indices=True to be able to invert)
        self.maxpool3d = nn.MaxPool3d(kernel_size=pool_kernel, stride=pool_stride, return_indices=True)
        self.maxunpool3d = nn.MaxUnpool3d(kernel_size=pool_kernel, stride=pool_stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C_in, H, W)

        Returns:
            Tensor of shape (B, C_out, H_out, W_out) where C_out = deconv_out_channels * D_out
            after folding the 3D depth into the channel dimension, L2-normalized across channels.
        """
        # 1) PixelUnshuffle: (B, C_in, H, W) -> (B, C_in * r^2, H/r, W/r)
        x_unshuffled = self.pixel_unshuffle(x)

        B, C_total, Hs, Ws = x_unshuffled.shape
        # Check that we can form the desired depth splits
        assert C_total % self.depth_splits == 0, (
            f"C_total ({C_total}) must be divisible by depth_splits ({self.depth_splits})."
        )
        C_per_depth = C_total // self.depth_splits

        # 2) Reshape to 5D: (B, C_per_depth, D, Hs, Ws)
        x5d = x_unshuffled.view(B, C_per_depth, self.depth_splits, Hs, Ws)

        # 3) LazyConvTranspose3d: operate on 5D tensors (B, C', D, H, W)
        deconv_out = self.deconv3d(x5d)  # lazy module will infer in_channels on first call

        # 4) MaxPool3d (with indices), then MaxUnpool3d to invert pooling
        pooled, indices = self.maxpool3d(deconv_out)
        unpooled = self.maxunpool3d(pooled, indices, output_size=deconv_out.size())

        # 5) Fold depth dimension into channels: (B, C_out, D_out, H_out, W_out) -> (B, C_out*D_out, H_out, W_out)
        B, C_out, D_out, H_out, W_out = unpooled.shape
        folded = unpooled.view(B, C_out * D_out, H_out, W_out)

        # 6) L2 normalization across channel dimension for stability
        denom = torch.norm(folded, p=2, dim=1, keepdim=True).clamp(min=1e-6)
        normalized = folded / denom

        return normalized


# Input configuration used by get_inputs()
B = BATCH_SIZE
C = IN_CHANNELS
H = HEIGHT
W = WIDTH

def get_inputs():
    """
    Returns a list containing a single input tensor shaped (BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH).
    """
    x = torch.randn(B, C, H, W)
    return [x]

def get_init_inputs():
    """
    Returns any initialization parameters required to construct the module externally.
    For this module we return the configuration constants so external test scaffolding
    can replicate the same settings if needed.
    """
    return [PIXEL_UNSHUFFLE_R, DEPTH_SPLITS, DECONV_OUT_CHANNELS, DECONV_KERNEL, DECONV_STRIDE, DECONV_PADDING, POOL_KERNEL, POOL_STRIDE]