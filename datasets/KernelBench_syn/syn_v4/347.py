import torch
import torch.nn as nn
from typing import List, Any

class Model(nn.Module):
    """
    Complex model that combines 1D constant padding, a 3D transposed convolution
    (interpreting the 1D signal as a 3D tensor with trivial spatial dims),
    and a SiLU non-linearity with a learned channel-wise gating pattern.

    The forward pass:
    1. Pads the input sequence in 1D using ConstantPad1d.
    2. Unsqueezes to 5D to apply ConvTranspose3d with kernel focused on the width dimension.
    3. Applies SiLU activation.
    4. Computes a channel-wise gate from the spatial mean and applies sigmoid gating.
    5. Reduces across channels to produce a 2D (batch, length) output.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_w: int = 4,
        stride_w: int = 2,
        conv_pad_w: int = 1,
        conv_out_pad: int = 0,
        pad_left: int = 1,
        pad_right: int = 1,
        pad_value: float = 0.0,
    ):
        """
        Initializes layers and parameters.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels for ConvTranspose3d.
            kernel_w (int): Kernel size along the width (1D length) dimension.
            stride_w (int): Stride along the width dimension.
            conv_pad_w (int): ConvTranspose3d padding along the width dimension.
            conv_out_pad (int): ConvTranspose3d output_padding along the width dimension.
            pad_left (int): Left padding for ConstantPad1d.
            pad_right (int): Right padding for ConstantPad1d.
            pad_value (float): Constant value to use for padding.
        """
        super(Model, self).__init__()
        # Save for potential introspection
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_w = kernel_w
        self.stride_w = stride_w
        self.conv_pad_w = conv_pad_w
        self.conv_out_pad = conv_out_pad
        self.pad_left = pad_left
        self.pad_right = pad_right
        self.pad_value = pad_value

        # 1D constant padding layer
        self.pad = nn.ConstantPad1d((pad_left, pad_right), pad_value)

        # ConvTranspose3d will operate on a shape (N, C, D=1, H=1, W=length)
        # Use kernel sizes that are trivial for D and H and focused on W
        self.deconv = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=(1, 1, kernel_w),
            stride=(1, 1, stride_w),
            padding=(0, 0, conv_pad_w),
            output_padding=(0, 0, conv_out_pad),
            bias=True,
        )

        # Non-linearity
        self.silu = nn.SiLU()

        # Small channel projection to produce a learnable scale for gating
        # This is optional but increases expressive power; implemented as 1x1x1 conv
        self.channel_proj = nn.ConvTranspose3d(
            out_channels, out_channels, kernel_size=(1, 1, 1), stride=1, padding=0, bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch, output_length),
                          produced by reducing across channels after gating.
        """
        # Step 1: 1D constant padding (preserves batch and channel dims)
        # Input: (N, C, L) -> Padded: (N, C, L + pad_left + pad_right)
        x_padded = self.pad(x)

        # Step 2: Interpret the 1D sequence as a 5D tensor for ConvTranspose3d:
        # (N, C, 1, 1, L_padded)
        x_5d = x_padded.unsqueeze(2).unsqueeze(3)

        # Step 3: Apply ConvTranspose3d to expand/transform along the width dimension
        # Output: (N, out_channels, 1, 1, L_out)
        y = self.deconv(x_5d)

        # Step 4: Squeeze trivial spatial dims -> (N, out_channels, L_out)
        y = y.squeeze(3).squeeze(2)

        # Step 5: Apply SiLU non-linearity
        y = self.silu(y)

        # Step 6: Produce a channel-wise modulation signal.
        # Compute spatial (length) mean per channel to summarize temporal activations:
        # gate_base shape: (N, out_channels, 1)
        gate_base = torch.mean(y, dim=2, keepdim=True)

        # Step 7: Optionally refine the gate with a small learned projection.
        # Use the channel_proj (1x1x1 ConvTranspose3d) by expanding dims, applying projection,
        # and squeezing back. This introduces a lightweight learned transform.
        gate_proj = self.channel_proj(gate_base.unsqueeze(2).unsqueeze(3))  # (N, outC, 1,1,1)
        gate_proj = gate_proj.squeeze(3).squeeze(2)  # (N, outC, 1)

        gate = torch.sigmoid(gate_proj)

        # Step 8: Apply gating to the activated features (channel-wise multiplicative gating)
        y_gated = y * gate

        # Step 9: Reduce across channels to produce final 2D output (batch, length)
        out = torch.sum(y_gated, dim=1)

        return out

# Configuration variables for generating inputs / initializing the model
batch_size = 8
in_channels = 16
out_channels = 24
seq_len = 64

# ConvTranspose3d kernel/stride/padding settings along width
kernel_w = 4
stride_w = 2
conv_pad_w = 1
conv_out_pad = 0

# ConstantPad1d settings
pad_left = 2
pad_right = 1
pad_value = 0.0

def get_inputs() -> List[torch.Tensor]:
    """
    Returns a list containing a single input tensor matching the expected
    input shape for the Model: (batch, in_channels, seq_len).
    """
    x = torch.randn(batch_size, in_channels, seq_len)
    return [x]

def get_init_inputs() -> List[Any]:
    """
    Returns the initialization parameters suitable for instantiating the Model.
    Order matches the Model.__init__ signature used for external initialization.
    """
    return [
        in_channels,
        out_channels,
        kernel_w,
        stride_w,
        conv_pad_w,
        conv_out_pad,
        pad_left,
        pad_right,
        pad_value,
    ]