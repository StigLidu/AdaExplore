import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex module that combines ConstantPad2d, LazyConvTranspose2d and Softmax to
    produce a channel-aware gated deconvolution pattern.

    Computation steps (forward):
    1. Constantly pad the input spatially.
    2. Apply a LazyConvTranspose2d (deconvolution) to produce an intermediate feature map A.
    3. Produce channel attention weights by global averaging A over spatial dims and applying Softmax over channels.
    4. Weight A by the channel attention, collapse channels to a single spatial attention map.
    5. Resize the spatial attention map to the padded input resolution, sigmoid-gate the padded input.
    6. Apply the same deconvolution to the gated input to produce a refined output.
    7. Return the residual sum of the initial deconvolution A and the refined output for stable gradients.
    """
    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        stride: int,
        conv_padding: int,
        output_padding: int,
        dilation: int,
        groups: int,
        pad_value: float,
    ):
        """
        Initializes the module components.

        Args:
            out_channels: number of output channels for the deconvolution.
            kernel_size: kernel size for the deconvolution.
            stride: stride for the deconvolution.
            conv_padding: padding parameter used by the deconvolution.
            output_padding: output_padding for the deconvolution.
            dilation: dilation for the deconvolution.
            groups: groups for the deconvolution.
            pad_value: constant value used by ConstantPad2d.
        """
        super(Model, self).__init__()
        # Constant padding around input; use the conv_padding as a convenient pad width
        pad_tuple = (conv_padding, conv_padding, conv_padding, conv_padding)
        self.pad = nn.ConstantPad2d(pad_tuple, pad_value)

        # Lazy deconvolution (in_channels is inferred at first forward)
        self.deconv = nn.LazyConvTranspose2d(
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=conv_padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=True,
        )

        # Softmax across channel dimension for attention
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, in_channels, H, W)

        Returns:
            Tensor: Refined feature tensor with shape determined by the deconvolution.
        """
        # 1) Pad input
        x_padded = self.pad(x)

        # 2) First deconvolution to produce intermediate features A
        A = self.deconv(x_padded)  # shape: (B, out_channels, H_a, W_a)

        # 3) Channel attention: global average spatially then softmax over channels
        # shape after mean: (B, out_channels, 1, 1)
        channel_summary = A.mean(dim=(2, 3), keepdim=True)
        channel_weights = self.softmax(channel_summary)  # normalized channel weights

        # 4) Weight A by channel_weights and collapse to single-channel spatial attention map
        A_weighted = A * channel_weights  # broadcast over spatial dims
        spatial_att = A_weighted.sum(dim=1, keepdim=True)  # (B, 1, H_a, W_a)

        # 5) Resize spatial attention to the padded input resolution and gate the input
        att_resized = F.interpolate(spatial_att, size=x_padded.shape[2:], mode='bilinear', align_corners=False)
        gate = torch.sigmoid(att_resized)  # values in (0,1)
        x_gated = x_padded * gate  # broadcast across channels

        # 6) Apply the same deconvolution to the gated input to get refined output
        refined = self.deconv(x_gated)

        # 7) Residual fusion (ensure shapes match between A and refined)
        # If shapes differ unexpectedly due to numerical rounding, interpolate refined or A to match.
        if A.shape != refined.shape:
            # Align refined to A's spatial dims for stable residual addition
            refined = F.interpolate(refined, size=A.shape[2:], mode='bilinear', align_corners=False)

        out = refined + A

        return out

# Configuration / default initialization parameters
batch_size = 8
in_channels = 3
height = 64
width = 64

out_channels = 16
kernel_size = 3
stride = 2
conv_padding = 1
output_padding = 1
dilation = 1
groups = 1
pad_value = -0.5

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [out_channels, kernel_size, stride, conv_padding, output_padding, dilation, groups, pad_value]