import torch
import torch.nn as nn
from typing import List

class Model(nn.Module):
    """
    Encoder-like 3D -> 2D transformation module that:
      - Applies a Lazy 3D convolution over a 5D input (N, C, D, H, W)
      - Reduces one spatial dimension via mean and adaptive average pooling (AdaptiveAvgPool1d)
      - Projects the pooled 1D sequence back into 2D and upsamples with a Lazy 2D transposed convolution
      - Final 1x1 Conv2d maps to desired output channels

    This demonstrates mixing nn.LazyConv3d, nn.AdaptiveAvgPool1d, and nn.LazyConvTranspose2d
    into a compact but nontrivial dataflow.
    """
    def __init__(
        self,
        mid_channels: int = 64,
        trans_out_channels: int = 32,
        pool_output: int = 8,
        out_channels: int = 3,
    ):
        """
        Args:
            mid_channels: number of output channels for the 3D conv
            trans_out_channels: number of output channels for the transposed 2D conv
            pool_output: target length for AdaptiveAvgPool1d (will become the width dim after reshaping)
            out_channels: number of channels for final 2D output
        """
        super(Model, self).__init__()
        # LazyConv3d will infer in_channels from the input tensor at first forward
        self.conv3d = nn.LazyConv3d(out_channels=mid_channels, kernel_size=3, padding=1)
        # Pool along a 1D length (we will prepare a (N, C, L) tensor before using it)
        self.pool1d = nn.AdaptiveAvgPool1d(output_size=pool_output)
        # LazyConvTranspose2d will infer in_channels at first forward
        # It upsamples spatial dims (we use kernel_size=3, stride=2 to enlarge height/width)
        self.convT2d = nn.LazyConvTranspose2d(out_channels=trans_out_channels,
                                              kernel_size=3, stride=2, padding=1, output_padding=1)
        # Final projection to requested output channels
        self.final_conv2d = nn.Conv2d(in_channels=trans_out_channels, out_channels=out_channels, kernel_size=1)
        # Activation
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (N, C_in, D, H, W)

        Returns:
            Tensor of shape (N, out_channels, H_out, W_out) where H_out ~ 2*H and W_out ~ 2*1 (due to design)
        """
        # x -> (N, mid_channels, D, H, W)
        x = self.conv3d(x)
        x = self.act(x)

        # Collapse the width dimension by averaging: (N, mid, D, H)
        x = x.mean(dim=4)

        # Rearrange to (N, mid, H, D) then flatten last two dims into a 1D sequence:
        # (N, mid, L) where L = H * D
        x = x.permute(0, 1, 3, 2).contiguous()
        N, Cmid, Hdim, Ddim = x.shape
        x = x.view(N, Cmid, Hdim * Ddim)

        # Adaptive 1D pooling to compress the sequence to the target length (pool_output)
        x = self.pool1d(x)  # (N, mid, pool_output)

        # Treat the pooled length as width dimension = 1 (we want a 2D map). Reshape to (N, mid, H, 1)
        # We choose H such that pool_output equals H (so width becomes 1). This keeps shapes clean.
        x = x.view(N, Cmid, Hdim, -1)  # (N, mid, H, 1) assuming pool_output == Hdim

        # Upsample / decode with transposed conv: (N, trans_out_channels, H*2, 2)
        x = self.convT2d(x)
        x = self.act(x)

        # Final 1x1 conv to desired output channels
        x = self.final_conv2d(x)
        return x

# Configuration / default input dimensions
BATCH = 4
IN_CHANNELS = 3
DEPTH = 12
HEIGHT = 8
WIDTH = 8

# Defaults used to construct the Model in tests or callers
DEFAULT_MID_CHANNELS = 64
DEFAULT_TRANS_OUT = 32
DEFAULT_POOL_OUTPUT = HEIGHT  # chosen to allow reshape (pool_output == original H)
DEFAULT_OUT_CHANNELS = IN_CHANNELS

def get_inputs():
    """
    Returns a list with a single input tensor shaped (BATCH, IN_CHANNELS, DEPTH, HEIGHT, WIDTH).
    """
    x = torch.randn(BATCH, IN_CHANNELS, DEPTH, HEIGHT, WIDTH)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters corresponding to the Model constructor defaults:
    [mid_channels, trans_out_channels, pool_output, out_channels]
    """
    return [DEFAULT_MID_CHANNELS, DEFAULT_TRANS_OUT, DEFAULT_POOL_OUTPUT, DEFAULT_OUT_CHANNELS]