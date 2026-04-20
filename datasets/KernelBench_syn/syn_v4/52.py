import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Any

class Model(nn.Module):
    """
    Complex model that combines a lazy 1D convolutional sequence encoder with 3D padding and Lp-pooling
    to produce spatially-aware channel gating for a volumetric input.

    Computation pattern:
    1. Encode a 1D sequence input with nn.LazyConv1d followed by ReLU.
    2. Global-average the conv output across the length dimension to produce channel-wise gates.
    3. Pad the 3D volume with nn.ZeroPad3d and apply nn.LPPool3d to aggregate local spatial energy.
    4. Upsample the pooled volume back to the original spatial resolution.
    5. Modulate the original volume with a gating factor derived from the sequence encoder and the pooled context:
       out = volume * (1 + sigmoid(gates) * upsampled_pooled)
    """
    def __init__(
        self,
        out_channels: int,
        conv_kernel_size: int,
        pool_norm_p: float,
        pool_kernel_size: Tuple[int, int, int],
        pad: Tuple[int, int, int, int, int, int],
    ):
        """
        Args:
            out_channels (int): Number of output channels from the LazyConv1d. Should match the volume channels.
            conv_kernel_size (int): Kernel size for the 1D convolution.
            pool_norm_p (float): The 'p' parameter for LPPool3d (e.g., 2.0 for L2 pooling).
            pool_kernel_size (tuple): Kernel size for LPPool3d across (D,H,W).
            pad (tuple): ZeroPad3d padding specified as (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back).
        """
        super(Model, self).__init__()
        # LazyConv1d will infer in_channels from the first forward pass of the sequence input.
        # Use padding to preserve sequence length roughly (same conv).
        padding = conv_kernel_size // 2
        self.seq_conv = nn.LazyConv1d(out_channels=out_channels, kernel_size=conv_kernel_size, padding=padding)
        self.activation = nn.ReLU(inplace=True)

        self.spatial_pad = nn.ZeroPad3d(pad)
        self.spatial_pool = nn.LPPool3d(norm_type=pool_norm_p, kernel_size=pool_kernel_size)

        # small learned scalar to control modulation strength (helps stability during initialization)
        self.modulation_scale = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def forward(self, x_seq: torch.Tensor, x_vol: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_seq (torch.Tensor): Sequence tensor of shape (N, C_seq, L).
            x_vol (torch.Tensor): Volume tensor of shape (N, C_vol, D, H, W).

        Returns:
            torch.Tensor: Modulated volume tensor of same shape as x_vol.
        """
        # Sequence branch: LazyConv1d -> ReLU -> global average over length -> sigmoid gates
        seq_feat = self.seq_conv(x_seq)                       # (N, out_channels, L)
        seq_feat = self.activation(seq_feat)                  # (N, out_channels, L)
        gates = seq_feat.mean(dim=2)                          # (N, out_channels)
        gates = torch.sigmoid(gates * self.modulation_scale)  # (N, out_channels)

        # Spatial branch: Pad -> LpPool3d -> upsample back to original spatial dims
        padded = self.spatial_pad(x_vol)                      # pad spatial dims
        pooled = self.spatial_pool(padded)                    # reduced spatial dims
        # Upsample pooled tensor to original D,H,W using trilinear interpolation
        target_size = (x_vol.shape[2], x_vol.shape[3], x_vol.shape[4])
        upsampled = F.interpolate(pooled, size=target_size, mode='trilinear', align_corners=False)

        # Reshape gates to broadcast over spatial dims and match channel dim
        gates = gates.view(x_vol.shape[0], x_vol.shape[1], 1, 1, 1)  # (N, C_vol, 1, 1, 1)

        # Modulate original volume using elementwise multiplication and a residual-like pattern
        out = x_vol * (1.0 + gates * upsampled)

        return out


# -------------------------
# Module-level configuration
# -------------------------
batch_size = 2
seq_channels = 8
seq_length = 128

vol_channels = 16
depth = 16
height = 32
width = 32

# Initialization parameters for the Model
out_channels = vol_channels         # ensure conv produces same number of channels as the volume
conv_kernel_size = 5
pool_norm_p = 2.0
pool_kernel_size = (2, 2, 2)
pad = (1, 1, 1, 1, 1, 1)             # ZeroPad3d expects 6-tuple padding


def get_inputs() -> List[torch.Tensor]:
    """
    Returns:
        List[torch.Tensor]: [x_seq, x_vol] where
            x_seq has shape (batch_size, seq_channels, seq_length)
            x_vol has shape (batch_size, vol_channels, depth, height, width)
    """
    x_seq = torch.randn(batch_size, seq_channels, seq_length)
    x_vol = torch.randn(batch_size, vol_channels, depth, height, width)
    return [x_seq, x_vol]


def get_init_inputs() -> List[Any]:
    """
    Returns:
        List[Any]: Initialization parameters for Model.__init__
    """
    return [out_channels, conv_kernel_size, pool_norm_p, pool_kernel_size, pad]