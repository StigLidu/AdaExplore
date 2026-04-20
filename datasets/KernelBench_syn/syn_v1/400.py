import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Complex module that:
- Applies AdaptiveMaxPool2d to spatially compress the input
- Rearranges pooled spatial dimensions into a 1D sequence and processes it with ConvTranspose1d
- Applies a learned PReLU activation
- Reshapes back to a 2D spatial tensor and computes a sigmoid gating mask
- Blends the transformed features with a channel-reduced version of the original input via the gating mask
This creates a cross-dimensional mixing pattern (2D -> 1D -> 2D) and a spatial gating/residual fusion.
"""

# Configuration variables
batch_size = 8
in_channels = 32
in_height = 64
in_width = 128

# Pooling target (reduces spatial resolution before 1D processing)
pool_H = 8
pool_W = 16

# ConvTranspose1d parameters (operates on the pooled width dimension)
conv_out_channels = 128  # Must be divisible by pool_H for reshaping back to 4D
conv_kernel_size = 5
conv_stride = 2
conv_padding = 1

class Model(nn.Module):
    """
    Model combining AdaptiveMaxPool2d, ConvTranspose1d, and PReLU with a gating/residual fusion.
    Input: (B, C, H, W)
    Output: (B, C_out, H_out, W_out) where:
      C_out = conv_out_channels // pool_H
      H_out = pool_H
      W_out = computed from ConvTranspose1d output length
    """
    def __init__(self):
        super(Model, self).__init__()

        # Validate reshape compatibility
        if conv_out_channels % pool_H != 0:
            raise ValueError("conv_out_channels must be divisible by pool_H for reshaping back to 4D.")
        if in_channels % (conv_out_channels // pool_H) != 0:
            # This restriction is to allow simple channel reduction by grouping the original input channels
            raise ValueError("in_channels must be divisible by conv_out_channels // pool_H to allow grouped reduction.")

        self.pool = nn.AdaptiveMaxPool2d((pool_H, pool_W))
        self.convtrans = nn.ConvTranspose1d(
            in_channels=in_channels * pool_H,
            out_channels=conv_out_channels,
            kernel_size=conv_kernel_size,
            stride=conv_stride,
            padding=conv_padding
        )
        # PReLU with per-channel parameters for the ConvTranspose output
        self.prelu = nn.PReLU(num_parameters=conv_out_channels)

        # Store some constants for use in forward
        self._pool_H = pool_H
        self._pool_W = pool_W
        self._channels_new = conv_out_channels // pool_H

        # Initialize ConvTranspose weights with a scaled normal for stability
        nn.init.normal_(self.convtrans.weight, mean=0.0, std=0.02)
        if self.convtrans.bias is not None:
            nn.init.zeros_(self.convtrans.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        1) Adaptive max pool to (pool_H, pool_W)
        2) Flatten channel and pooled-height dims to create a (B, C * pool_H, pool_W) 1D sequence per example
        3) Apply ConvTranspose1d to mix across that 1D sequence and change channels
        4) Apply PReLU
        5) Reshape back to 4D: (B, channels_new, pool_H, W_out)
        6) Compute a sigmoid gate from the transformed features (spatially aggregated per-channel)
        7) Reduce the original input's channels by grouping and averaging to match channels_new and spatial size
        8) Blend transformed features and reduced original via the gate and apply a final ReLU
        """
        # x: (B, C, H, W)
        B, C, H, W = x.shape

        # 1) Spatial pooling
        pooled = self.pool(x)  # (B, C, pool_H, pool_W)

        # 2) Rearrange for 1D conv: (B, C * pool_H, pool_W)
        Bp, Cp, Hp, Wp = pooled.shape  # Hp should equal self._pool_H, Wp == self._pool_W
        seq = pooled.view(Bp, Cp * Hp, Wp)

        # 3) 1D transposed convolution
        conv_out = self.convtrans(seq)  # (B, conv_out_channels, L_out)

        # 4) Learnable nonlinearity
        conv_out = self.prelu(conv_out)

        # 5) Reshape back to 4D
        L_out = conv_out.shape[2]
        channels_new = self._channels_new
        H_new = self._pool_H
        W_new = L_out
        # conv_out has shape (B, conv_out_channels, L_out) -> view to (B, channels_new, H_new, W_new)
        transformed = conv_out.view(B, channels_new, H_new, W_new)

        # 6) Compute a spatial gating mask (collapsed over channels to get a single-channel gate)
        gate = torch.sigmoid(transformed.mean(dim=1, keepdim=True))  # (B, 1, H_new, W_new)

        # 7) Reduce original input to match (B, channels_new, H_new, W_new) via adaptive avg pooling and grouped averaging
        # First resize original spatially
        orig_spatial = F.adaptive_avg_pool2d(x, (H_new, W_new))  # (B, C, H_new, W_new)

        # Group channels: in_channels must be divisible by channels_new
        group_size = C // channels_new
        orig_grouped = orig_spatial.view(B, channels_new, group_size, H_new, W_new)
        orig_reduced = orig_grouped.mean(dim=2)  # (B, channels_new, H_new, W_new)

        # 8) Blend transformed features and reduced original using the gate, then final activation
        out = transformed * gate + orig_reduced * (1.0 - gate)
        out = F.relu(out)

        return out

def get_inputs():
    """
    Returns a sample input tensor matching the configuration above.
    """
    x = torch.randn(batch_size, in_channels, in_height, in_width)
    return [x]

def get_init_inputs():
    """
    No external initialization parameters required for this model.
    """
    return []