import torch
import torch.nn as nn
from typing import List

class Model(nn.Module):
    """
    Complex 3D -> 2D projection model that:
      - Upsamples volumetric data with a lazy ConvTranspose3d (lazy in_channels).
      - Applies 2D channel-wise dropout and an Lp pooling on the spatial planes per-slice.
      - Projects channels with a 1x1x1 Conv3d and reduces along depth to produce 2D feature maps.

    This model demonstrates interplay between 3D and 2D ops by reshaping/permuting tensors
    so that Dropout2d and LPPool2d operate on (N, C, H, W) inputs corresponding to each
    depth slice across the batch.
    """
    def __init__(self, deconv_out_channels: int = 32, proj_out_channels: int = 16, dropout_p: float = 0.2):
        """
        Args:
            deconv_out_channels: number of output channels for the ConvTranspose3d.
            proj_out_channels: number of output channels after 1x1x1 Conv3d projection.
            dropout_p: probability for Dropout2d.
        """
        super(Model, self).__init__()
        # LazyConvTranspose3d will infer in_channels at first forward call
        # kernel/stride chosen to perform a modest upsampling in D, H, W
        self.deconv = nn.LazyConvTranspose3d(
            out_channels=deconv_out_channels,
            kernel_size=(3, 4, 4),
            stride=(2, 2, 2),
            padding=(1, 1, 1),
            output_padding=(1, 0, 0),
            bias=True
        )
        # Channel-wise dropout for 2D slices
        self.dropout2d = nn.Dropout2d(p=dropout_p)
        # Lp pooling over HxW for each (batch, channel) 2D slice
        self.lp_pool = nn.LPPool2d(norm_type=2, kernel_size=2, stride=2)
        # 1x1x1 projection to reduce channel dimensionality in 3D tensor
        self.proj_conv = nn.Conv3d(deconv_out_channels, proj_out_channels, kernel_size=1, bias=True)
        # Non-linearity after projection
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input volumetric tensor of shape (B, C_in, D, H, W)

        Returns:
            Tensor of shape (B, proj_out_channels, H_out, W_out) after depth-averaged projection.
        """
        # 1) Upsample / transpose-convolution in 3D
        x = self.deconv(x)  # -> (B, C_deconv, D1, H1, W1)

        B, C, D1, H1, W1 = x.shape

        # 2) Reorder and merge batch & depth dims so we can apply 2D ops per-slice:
        #    (B, C, D1, H1, W1) -> (B, D1, C, H1, W1) -> (B*D1, C, H1, W1)
        x_slices = x.permute(0, 2, 1, 3, 4).contiguous().view(B * D1, C, H1, W1)

        # 3) Channel-wise dropout across each 2D slice
        x_slices = self.dropout2d(x_slices)

        # 4) Lp pooling over HxW reduces spatial resolution for each slice
        x_slices = self.lp_pool(x_slices)  # -> (B*D1, C, H2, W2)

        # Recover batched 3D layout: (B*D1, C, H2, W2) -> (B, D1, C, H2, W2) -> (B, C, D1, H2, W2)
        _, _, H2, W2 = x_slices.shape
        x_3d = x_slices.view(B, D1, C, H2, W2).permute(0, 2, 1, 3, 4).contiguous()

        # 5) 1x1x1 projection (channel reduction) in 3D
        x_3d = self.proj_conv(x_3d)  # -> (B, proj_out_channels, D1, H2, W2)

        # 6) Non-linearity and final reduction: average across depth to produce 2D feature maps
        x_3d = self.act(x_3d)
        out = x_3d.mean(dim=2)  # -> (B, proj_out_channels, H2, W2)

        return out

# Configuration / default parameters for generating inputs
batch_size = 4
in_channels = 8
depth = 4
height = 32
width = 32

deconv_out_channels = 32
proj_out_channels = 16
dropout_p = 0.2

def get_inputs() -> List[torch.Tensor]:
    """
    Create a random 5D tensor suitable for the model:
      (batch_size, in_channels, depth, height, width)
    """
    x = torch.randn(batch_size, in_channels, depth, height, width, dtype=torch.float32)
    return [x]

def get_init_inputs() -> List:
    """
    Return the initialization parameters for the Model constructor in the same order.
    """
    return [deconv_out_channels, proj_out_channels, dropout_p]