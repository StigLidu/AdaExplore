import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex image patch attention module that:
    - Uses PixelUnshuffle to decompose spatial resolution into channel bundles
    - Extracts overlapping patches (via F.unfold)
    - Computes channel-wise attention per patch using Softmax
    - Aggregates patches and reconstructs a low-resolution attention map via nn.Fold
    - Upsamples the attention map and applies it as a spatial-channel-aware gating to the original input

    This pattern combines nn.PixelUnshuffle, nn.Softmax, and nn.Fold with standard tensor ops.
    """
    def __init__(self, downscale: int, kernel_size: int, height: int, width: int, in_channels: int):
        """
        Args:
            downscale (int): PixelUnshuffle downscale factor (must divide height and width).
            kernel_size (int): Patch kernel size for unfolding/folding (preferably odd).
            height (int): Input height (used to configure Fold output size).
            width (int): Input width (used to configure Fold output size).
            in_channels (int): Number of channels in the input image.
        """
        super(Model, self).__init__()
        assert height % downscale == 0 and width % downscale == 0, "height and width must be divisible by downscale"
        self.downscale = downscale
        self.kernel_size = kernel_size
        self.in_channels = in_channels

        # PixelUnshuffle to create channel bundles of size in_channels * downscale^2
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale)

        # Softmax will compute attention across the channel-bundle dimension
        # We will arrange tensors so that dim=1 corresponds to the channel-bundle axis when applying Softmax
        self.softmax_channel = nn.Softmax(dim=1)

        # Compute reduced spatial dimensions after PixelUnshuffle
        self.h_small = height // downscale
        self.w_small = width // downscale

        # nn.Fold to reconstruct a single-channel attention map from aggregated patches
        pad = kernel_size // 2  # keep same spatial size with padding
        self.fold = nn.Fold(output_size=(self.h_small, self.w_small),
                            kernel_size=(kernel_size, kernel_size),
                            padding=pad,
                            stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
            x: (B, C, H, W)
        Returns:
            gated: (B, C, H, W) original image gated by the learned attention map
        """
        # 1) PixelUnshuffle: (B, C, H, W) -> (B, C * r^2, H/r, W/r)
        x_small = self.pixel_unshuffle(x)
        B, C2, Hs, Ws = x_small.shape  # C2 = in_channels * downscale^2

        # 2) Extract overlapping patches from the lower-resolution tensor
        #    patches: (B, C2 * k*k, L) where L = Hs * Ws (with stride=1 and same padding)
        patches = F.unfold(x_small, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)
        # reshape to (B, C2, k*k, L) to isolate the channel-bundle axis
        k2 = self.kernel_size * self.kernel_size
        patches_reshaped = patches.view(B, C2, k2, -1)  # (-1) == L

        # 3) Compute channel-wise attention per patch position using Softmax across C2
        #    softmax applies across dim=1 (C2) for each (B, k*k, L)
        channel_attention = self.softmax_channel(patches_reshaped)

        # 4) Weighted aggregation across channel-bundles to produce a single-value per kernel position per patch
        #    weighted_patches: (B, k*k, L)
        weighted_patches = (patches_reshaped * channel_attention).sum(dim=1)

        # 5) Fold aggregated patch values back into a single-channel low-resolution map
        #    fold expects (B, C_out * k*k, L). We want C_out = 1, so input shape is (B, k*k, L)
        folded = self.fold(weighted_patches)  # -> (B, 1, Hs, Ws)

        # 6) Upsample the low-res attention map to original resolution
        attn_up = F.interpolate(folded, scale_factor=self.downscale, mode='nearest')  # -> (B, 1, H, W)

        # 7) Normalize attention to [0,1] via sigmoid, and apply as multiplicative gating to the original input
        attn_map = torch.sigmoid(attn_up)
        gated = x * attn_map  # broadcast across channels

        # 8) Add a light residual to retain some original content (stabilizes gradients)
        #    Combine gated and a small fraction of original input
        out = 0.9 * gated + 0.1 * x

        return out

# Module-level configuration (used by get_inputs / get_init_inputs)
batch_size = 8
in_channels = 3
height = 64
width = 64
downscale = 2   # PixelUnshuffle factor (must divide height and width)
kernel_size = 3  # patch size for unfold/fold (odd recommended)

def get_inputs():
    """
    Returns a list with a single input tensor for the model's forward method.
    Shape: (batch_size, in_channels, height, width)
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns the list of initialization arguments to create the Model instance in order:
    [downscale, kernel_size, height, width, in_channels]
    """
    return [downscale, kernel_size, height, width, in_channels]