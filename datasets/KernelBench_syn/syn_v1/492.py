import torch
import torch.nn as nn
from typing import List

class Model(nn.Module):
    """
    Complex 1D upsampling module that demonstrates:
      - Zero padding of inputs
      - Two parallel ConvTranspose1d paths with different receptive fields
      - Non-linear gating using Tanh on one path
      - Element-wise interaction (multiplicative gating) between paths
      - A 1x "projection" ConvTranspose1d to map to desired output channels
      - Final Tanh activation for bounded outputs

    The forward pass:
      1. Zero-pad the input in time dimension.
      2. Send padded input through a larger transposed convolution (path A).
      3. Send the original (unpadded) input through a smaller transposed convolution (path B).
      4. Apply Tanh to path B to form a gating signal.
      5. Element-wise multiply path A and gated path B (after aligning temporal lengths).
      6. Project the combined representation to out_channels using a 1x ConvTranspose1d.
      7. Apply final Tanh and return.
    """
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        pad_amount: int
    ):
        super(Model, self).__init__()
        # ZeroPad1d pads last dimension (time) with pad_amount on both sides
        self.pad = nn.ZeroPad1d(pad_amount)

        # Primary transposed convolution path (larger receptive field / upsampling)
        self.deconv_a = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )

        # Secondary transposed convolution path (smaller receptive field)
        # Choose smaller kernel/stride to create a different temporal footprint
        kernel_small = max(1, kernel_size // 2)
        stride_small = max(1, stride // 2)
        padding_small = max(0, padding - 1)
        self.deconv_b = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel_small,
            stride=stride_small,
            padding=padding_small,
            bias=True
        )

        # Projection conv to map combined mid_channels -> out_channels
        # Use kernel_size=1 to mix channels without changing temporal dim
        self.proj = nn.ConvTranspose1d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )

        # Non-linearities
        self.gate_act = nn.Tanh()
        self.final_act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation.

        Args:
            x (torch.Tensor): Input tensor with shape (batch, in_channels, time_len)

        Returns:
            torch.Tensor: Output tensor with shape (batch, out_channels, new_time_len)
        """
        # 1) Pad the input in time dimension
        x_padded = self.pad(x)

        # 2) Larger-path transposed convolution
        y_a = self.deconv_a(x_padded)  # shape: (batch, mid_channels, L_a)

        # 3) Smaller-path transposed convolution (on original signal)
        y_b = self.deconv_b(x)         # shape: (batch, mid_channels, L_b)

        # 4) Apply gating non-linearity to path B
        gate = self.gate_act(y_b)

        # 5) Align temporal lengths: crop to the minimum length so elementwise ops are valid
        len_a = y_a.size(2)
        len_gate = gate.size(2)
        min_len = min(len_a, len_gate)
        if min_len <= 0:
            # defensive: if something degenerated, return zeros with out_channels
            batch = x.size(0)
            return torch.zeros(batch, self.proj.out_channels, 0, dtype=x.dtype, device=x.device)

        y_a_cropped = y_a[..., :min_len]
        gate_cropped = gate[..., :min_len]

        # 6) Element-wise multiplicative interaction (gating of path A by path B)
        combined = y_a_cropped * gate_cropped  # (batch, mid_channels, min_len)

        # 7) Project combined representation to desired out_channels and apply final non-linearity
        out = self.proj(combined)
        out = self.final_act(out)
        return out

# Configuration / default initialization parameters
batch_size = 8
in_channels = 16
mid_channels = 32
out_channels = 12
kernel_size = 4
stride = 2
padding = 1
pad_amount = 2
time_length = 256

def get_inputs() -> List[torch.Tensor]:
    """
    Create a standard input tensor for the model.

    Returns:
        List[torch.Tensor]: single-element list with input tensor of shape
                            (batch_size, in_channels, time_length)
    """
    x = torch.randn(batch_size, in_channels, time_length)
    return [x]

def get_init_inputs() -> List[int]:
    """
    Return the initialization parameters used to construct the Model instance.

    Order matches Model.__init__ signature:
      (in_channels, mid_channels, out_channels, kernel_size, stride, padding, pad_amount)
    """
    return [in_channels, mid_channels, out_channels, kernel_size, stride, padding, pad_amount]