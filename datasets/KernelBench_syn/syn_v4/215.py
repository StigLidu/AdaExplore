import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

class Model(nn.Module):
    """
    Complex 3D processing model that:
      - Applies AdaptiveAvgPool3d to produce a compact spatial representation.
      - Applies a LazyConvTranspose3d to perform learned upsampling / channel transform.
      - Converts 3D feature maps into 1D sequences and applies AdaptiveMaxPool1d.
      - Fuses global channel descriptors with pooled sequences and redistributes attention
        back onto the spatial map via interpolation and elementwise gating.

    This chain creates a non-trivial flow mixing pooling, transpose-conv, 1D pooling,
    and sequence-based attention redistribution.
    """
    def __init__(self,
                 avg_output_size: Tuple[int, int, int],
                 up_out_channels: int,
                 up_kernel_size: int,
                 up_stride: int,
                 seq_output_size: int):
        """
        Args:
            avg_output_size: output spatial size for AdaptiveAvgPool3d (od, oh, ow).
            up_out_channels: number of output channels for the LazyConvTranspose3d.
            up_kernel_size: kernel size for the transpose convolution.
            up_stride: stride for the transpose convolution.
            seq_output_size: output length for AdaptiveMaxPool1d applied to flattened spatial sequence.
        """
        super(Model, self).__init__()
        # 3D adaptive average pooling to compress spatial information
        self.avgpool3d = nn.AdaptiveAvgPool3d(output_size=avg_output_size)
        # Lazy transpose conv to learn an upsampling/channel transform (in_channels inferred at first forward)
        self.upconv3d = nn.LazyConvTranspose3d(out_channels=up_out_channels,
                                               kernel_size=up_kernel_size,
                                               stride=up_stride,
                                               padding=up_kernel_size // 2)
        # 1D adaptive max pooling to condense spatial sequence representations
        self.adaptive_max1d = nn.AdaptiveMaxPool1d(output_size=seq_output_size)
        # small epsilon for numerical stability when doing scaling ops
        self.eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor with shape (batch, in_channels, D, H, W)

        Returns:
            A tensor representing gated, redistributed spatial features.
            Shape will be (batch, up_out_channels, od_up, oh_up, ow_up)
            where od_up/oh_up/ow_up depend on upconv params and the avgpool output.
        """
        # Step 1: Compress spatially with adaptive average pooling
        # x_avg: (B, C_in, od, oh, ow)
        x_avg = self.avgpool3d(x)

        # Step 2: Non-linearity
        x_avg = torch.relu(x_avg)

        # Step 3: Learned transpose convolution to change channels and spatial resolution
        # x_up: (B, C_up, od_up, oh_up, ow_up)
        x_up = self.upconv3d(x_avg)

        # Step 4: Global per-channel descriptor from upsampled feature map
        # desc: (B, C_up)
        desc = x_up.mean(dim=[2, 3, 4])

        # Step 5: Flatten spatial dims into a sequence for 1D pooling
        B, C_up, od_up, oh_up, ow_up = x_up.shape
        L = od_up * oh_up * ow_up
        # x_seq: (B, C_up, L)
        x_seq = x_up.view(B, C_up, L)

        # Step 6: Adaptive max pooling on the sequence dimension -> (B, C_up, seq_len)
        seq_pooled = self.adaptive_max1d(x_seq)

        # Step 7: Fuse global descriptor with pooled sequence using a channel gating mechanism
        # desc_unsq: (B, C_up, 1) -> broadcast across sequence length
        desc_unsq = desc.unsqueeze(-1)
        # gating pre-activation: combine multiplicatively and add a small epsilon then sigmoid
        gating = torch.sigmoid(seq_pooled * desc_unsq / (desc_unsq.abs() + self.eps))

        # Step 8: Upsample gating from seq_len back to L (original spatial sequence length)
        # Use linear interpolation along the sequence axis
        gating_up = F.interpolate(gating, size=L, mode='linear', align_corners=False)

        # Step 9: Re-weight the original upsampled spatial map by the gating and reshape back to 3D
        gated_seq = x_seq * gating_up  # (B, C_up, L)
        out = gated_seq.view(B, C_up, od_up, oh_up, ow_up)

        # Final non-linearity
        out = torch.relu(out)

        return out

# -------------------------
# Module-level configuration (example sizes)
# -------------------------
batch_size = 8
in_channels = 3
depth = 16
height = 32
width = 32

# AdaptiveAvgPool3d target output size (od, oh, ow)
avg_output_size = (4, 8, 8)

# LazyConvTranspose3d parameters
up_out_channels = 16
up_kernel_size = 3
up_stride = 2

# AdaptiveMaxPool1d output length for the sequence pooling
seq_output_size = 16

def get_inputs() -> List[torch.Tensor]:
    """
    Create input tensors for the model.

    Returns:
        A list containing a single 5D input tensor with shape (batch_size, in_channels, depth, height, width).
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs() -> List:
    """
    Return initialization parameters for the Model constructor in the order:
      [avg_output_size, up_out_channels, up_kernel_size, up_stride, seq_output_size]
    """
    return [avg_output_size, up_out_channels, up_kernel_size, up_stride, seq_output_size]