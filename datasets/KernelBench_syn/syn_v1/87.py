import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class Model(nn.Module):
    """
    A moderately complex 1D processing module that demonstrates a small
    depthwise convolutional block with replication padding, channel mixing,
    layer normalization, and a gated residual path using LeakyReLU.

    Pipeline:
    1. ReplicationPad1d to extend boundaries.
    2. Depthwise Conv1d (groups=in_channels) to apply per-channel filters.
    3. Pointwise Conv1d (1x1) to mix channel information.
    4. Transpose -> LayerNorm over channels -> LeakyReLU.
    5. Position-wise feed-forward (Linear) applied to last dim.
    6. Residual connection with length-adaptive alignment (slice or replicate-pad).
    """
    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        padding: int = 2,
        negative_slope: float = 0.01,
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            kernel_size (int): Width of the depthwise kernel.
            stride (int): Stride for the depthwise convolution.
            padding (int): Amount of replication padding applied on both sides.
            negative_slope (float): Negative slope for LeakyReLU.
        """
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Replicate boundary values to provide stable padded input
        self.pad = nn.ReplicationPad1d(self.padding)

        # Depthwise conv: one filter per input channel (groups=in_channels)
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=0,  # padding handled by ReplicationPad1d
            groups=in_channels,
            bias=False
        )

        # Pointwise conv to allow channel mixing after depthwise filtering
        self.pointwise = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=True)

        # LayerNorm applied over channel dimension -> operate on (batch, seq, channels)
        self.norm = nn.LayerNorm(in_channels)

        # Non-linearity
        self.act = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

        # Position-wise feed-forward (applied to last dim after transpose)
        self.ff = nn.Linear(in_channels, in_channels, bias=True)

    def _align_residual(self, residual: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        Aligns the residual tensor along the temporal dimension to match target_len.
        If residual is longer -> slice. If shorter -> replicate-pad on the right.
        """
        res_len = residual.size(2)
        if res_len == target_len:
            return residual
        elif res_len > target_len:
            # chop off extra timesteps
            return residual[:, :, :target_len]
        else:
            # pad on the right using replication to match the length
            pad_amount = target_len - res_len
            # F.pad expects (pad_left, pad_right) for last dim
            return F.pad(residual, (0, pad_amount), mode='replicate')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor with shape (batch, channels, length)

        Returns:
            torch.Tensor: Output tensor with same channel dimension and adjusted length
                          depending on stride/padding settings.
        """
        # Save residual for later
        residual = x

        # 1) Replication padding to mitigate edge artifacts
        x = self.pad(x)  # (B, C, L + 2*padding)

        # 2) Depthwise convolution (per-channel filtering)
        x = self.depthwise(x)  # (B, C, L_out)

        # 3) Channel mixing via pointwise conv
        x = self.pointwise(x)  # (B, C, L_out)

        # 4) Normalize per position over channels (LayerNorm expects (B, L, C))
        x = x.transpose(1, 2)  # (B, L_out, C)
        x = self.norm(x)       # LayerNorm over last dim (channels)

        # 5) Non-linearity
        x = self.act(x)

        # 6) Position-wise feed-forward (applied along channels at each timestep)
        x = self.ff(x)         # (B, L_out, C)

        # 7) Restore to (B, C, L_out)
        x = x.transpose(1, 2)

        # 8) Align residual to match temporal length and combine
        target_len = x.size(2)
        residual_aligned = self._align_residual(residual, target_len)
        out = x + residual_aligned

        # 9) Small scaling to stabilize magnitudes
        return out * 0.5


# Module-level configuration variables
batch_size = 8
in_channels = 16
seq_length = 64
kernel_size = 5
stride = 1
padding = kernel_size // 2  # common choice to preserve length when stride=1
negative_slope = 0.02

def get_inputs() -> List[torch.Tensor]:
    """
    Returns a list containing a single randomized input tensor matching the
    configured batch_size, in_channels, and seq_length.

    Shape: (batch_size, in_channels, seq_length)
    """
    x = torch.randn(batch_size, in_channels, seq_length)
    return [x]

def get_init_inputs() -> List:
    """
    Returns a list of initialization parameters for the Model constructor
    in the order: in_channels, kernel_size, stride, padding, negative_slope
    """
    return [in_channels, kernel_size, stride, padding, negative_slope]