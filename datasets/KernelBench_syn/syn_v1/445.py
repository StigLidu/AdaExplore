import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex 1D upsampling module using stacked ConvTranspose1d layers with learnable PReLU
    activations and a residual shortcut. The model performs two staged transposed convolutions
    to increase the temporal resolution by a factor of 4 (scale 2 then scale 2), applies PReLU
    after each transposed convolution, adds a projected shortcut (input is nearest-neighbor
    upsampled and channel-projected via a 1x1 ConvTranspose1d), and finally clamps outputs with Hardtanh.

    Forward computation:
        x -> convt1 -> prelu1 -> convt2 -> prelu2 -> (residual add from upsampled+projected input) -> hardtanh -> out

    This structure demonstrates combining ConvTranspose1d, PReLU and Hardtanh into a coherent block
    with differing tensor shapes and a residual path.
    """
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel1: int,
        stride1: int,
        padding1: int,
        output_padding1: int,
        kernel2: int,
        stride2: int,
        padding2: int,
        output_padding2: int,
    ):
        """
        Initializes the transposed convolution layers, activations, shortcut projection, and clamp.

        Args:
            in_channels (int): Number of channels in the input.
            mid_channels (int): Number of channels after the first ConvTranspose1d.
            out_channels (int): Number of output channels after the second ConvTranspose1d.
            kernel1/stride1/padding1/output_padding1: Parameters for the first ConvTranspose1d.
            kernel2/stride2/padding2/output_padding2: Parameters for the second ConvTranspose1d.
        """
        super(Model, self).__init__()

        # First stage upsamples by stride1 (e.g., 2) and increases channels to mid_channels
        self.convt1 = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel1,
            stride=stride1,
            padding=padding1,
            output_padding=output_padding1,
        )

        # Second stage upsamples by stride2 (e.g., 2) and produces out_channels
        self.convt2 = nn.ConvTranspose1d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=kernel2,
            stride=stride2,
            padding=padding2,
            output_padding=output_padding2,
        )

        # Shortcut projection: 1x1 ConvTranspose1d (no temporal change) to map input channels to out_channels
        self.shortcut_conv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            output_padding=0,
        )

        # Learnable, channel-wise PReLU activations
        self.prelu1 = nn.PReLU(num_parameters=mid_channels)
        self.prelu2 = nn.PReLU(num_parameters=out_channels)

        # Final clamping to a stable range
        self.clamp = nn.Hardtanh(min_val=-1.0, max_val=1.0, inplace=False)

        # Initialize weights for stable behavior (Xavier for transposed convs, zeros for biases)
        for m in (self.convt1, self.convt2, self.shortcut_conv):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length * upsample_factor).
        """
        # Stage 1: upsample channels -> mid_channels
        out = self.convt1(x)         # shape: (N, mid_channels, L * stride1)
        out = self.prelu1(out)       # activation

        # Stage 2: upsample to final out_channels
        out = self.convt2(out)       # shape: (N, out_channels, L * stride1 * stride2)
        out = self.prelu2(out)

        # Residual shortcut: nearest-neighbor upsample input to match temporal length, then project channels
        # Use scale factor equal to total upsampling of the stacked transposed convs.
        # Compute scale_factor from lengths (we infer using integer math): rely on configured strides in get_init_inputs
        # Here we assume forward caller provided input compatible with configured strides.
        # Nearest interpolation preserves the temporal structure for the residual.
        scale_factor = (stride1_global * stride2_global)  # module-level globals defined below
        shortcut = F.interpolate(x, scale_factor=scale_factor, mode='nearest')  # shape: (N, in_channels, L * scale_factor)
        shortcut = self.shortcut_conv(shortcut)  # project channels to out_channels

        # Add residual and clamp
        out = out + shortcut
        out = self.clamp(out)
        return out

# Module-level configuration variables (used by get_inputs and get_init_inputs)
batch_size = 8
in_channels_global = 16
mid_channels_global = 32
out_channels_global = 8
length = 64

# Parameters for the two ConvTranspose1d stages (chosen to produce exact upsample by 2 each)
kernel1 = 4
stride1_global = 2
padding1 = 1
output_padding1 = 0

kernel2 = 4
stride2_global = 2
padding2 = 1
output_padding2 = 0

def get_inputs():
    """
    Returns a list containing the input tensor for the model. The tensor has shape:
    (batch_size, in_channels_global, length)
    """
    x = torch.randn(batch_size, in_channels_global, length)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters that match the Model constructor order:
    in_channels, mid_channels, out_channels,
    kernel1, stride1, padding1, output_padding1,
    kernel2, stride2, padding2, output_padding2
    """
    return [
        in_channels_global,
        mid_channels_global,
        out_channels_global,
        kernel1,
        stride1_global,
        padding1,
        output_padding1,
        kernel2,
        stride2_global,
        padding2,
        output_padding2,
    ]