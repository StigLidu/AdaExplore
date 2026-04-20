import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Any

# Configuration
batch_size = 4
in_channels = 8
mid_channels = 16
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
groups = 4  # must divide mid_channels
input_length = 64  # temporal length of the 1D input
upsample_scale = (2, 1)  # will be used by UpsamplingNearest2d on (H, W)

class Model(nn.Module):
    """
    A composite module that demonstrates a temporal upsampling pipeline using:
      - ConvTranspose1d (temporal / 1D transposed convolution)
      - GroupNorm (channel-wise group normalization for 1D tensors)
      - Non-linear activation (GELU)
      - Conversion to 2D-like tensor and UpsamplingNearest2d to further upsample the temporal axis

    The forward pass:
      1. Applies ConvTranspose1d to double the temporal resolution (due to chosen stride/padding/output_padding).
      2. Normalizes channels with GroupNorm.
      3. Applies GELU activation.
      4. Adds a singleton spatial dimension to treat the 1D signal as a (H, W=1) image.
      5. Upsamples the "height" (temporal axis) with nearest-neighbor 2D upsampling.
      6. Collapses the singleton width dimension to return a (N, C, L_out) tensor.
    """
    def __init__(
        self,
        in_channels: int = in_channels,
        mid_channels: int = mid_channels,
        kernel_size: int = kernel_size,
        stride: int = stride,
        padding: int = padding,
        output_padding: int = output_padding,
        groups: int = groups,
        upsample_scale: Tuple[int, int] = upsample_scale
    ):
        super(Model, self).__init__()
        # Transposed conv to increase temporal resolution (1D)
        self.deconv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=True
        )
        # Group normalization over the channel dimension (works with 1D input shape N, C, L)
        self.gn = nn.GroupNorm(num_groups=groups, num_channels=mid_channels)
        # Nearest-neighbor 2D upsampling. We'll treat the 1D signal as H x W=1 and upsample H.
        self.upsample = nn.UpsamplingNearest2d(scale_factor=upsample_scale)
        # Small final linear projection implemented as a 1x1 convolution in 1D (Conv1d with kernel_size=1)
        # This projects back to the same number of channels but could be adjusted if needed.
        self.project = nn.Conv1d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C_in, L_in)

        Returns:
            torch.Tensor: Output tensor of shape (N, C_mid, L_out)
        """
        # 1) Transposed convolution: increases temporal length (L_out = 2 * L_in with our hyperparams)
        x = self.deconv(x)  # shape -> (N, mid_channels, L_deconv)

        # 2) Group normalization across channels
        x = self.gn(x)  # shape preserved

        # 3) Non-linearity
        x = F.gelu(x)

        # 4) Treat temporal dimension as spatial height: (N, C, H=L_deconv, W=1)
        x = x.unsqueeze(-1)

        # 5) 2D nearest-neighbor upsampling (doubles the height if upsample_scale=(2,1))
        x = self.upsample(x)  # shape -> (N, C, H_up, W=1)

        # 6) Collapse the singleton width dimension back to 1D: (N, C, L_final)
        x = x.squeeze(-1)

        # 7) Final 1x1 projection (channel-wise mixing)
        x = self.project(x)

        return x

def get_inputs() -> List[torch.Tensor]:
    """
    Create a batch of random 1D signals for the model.

    Returns:
        list: [x] where x has shape (batch_size, in_channels, input_length)
    """
    x = torch.randn(batch_size, in_channels, input_length)
    return [x]

def get_init_inputs() -> List[Any]:
    """
    Provides the initialization arguments for the Model constructor in the same order.

    Returns:
        list: Constructor args for Model(...)
    """
    return [
        in_channels,
        mid_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        groups,
        upsample_scale
    ]