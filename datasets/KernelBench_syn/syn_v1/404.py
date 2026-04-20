import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class Model(nn.Module):
    """
    Complex 1D upsampling module that uses a lazy transposed convolution followed by
    nonlinear gating. The module demonstrates:
      - LazyConvTranspose1d (in_channels inferred at first forward)
      - Hardtanh activation for clipping
      - Hardsigmoid used to create a channel-wise gate from global statistics
      - A learnable channel-wise scale parameter applied to the final pooled output

    Forward computation steps:
      1. Apply LazyConvTranspose1d to input (upsampling / transpose conv)
      2. Apply Hardtanh nonlinearity
      3. Compute channel-wise global average over time
      4. Produce a channel-wise gate via Hardsigmoid
      5. Modulate the conv output with the gate and reduce over time dimension
      6. Apply a learned channel-wise scaling factor and return (batch, out_channels)
    """
    def __init__(self, out_channels: int, kernel_size: int = 4, stride: int = 2, padding: int = 1):
        """
        Initializes the model.

        Args:
            out_channels (int): Number of output channels produced by the transposed convolution.
            kernel_size (int): Kernel size for the transposed convolution.
            stride (int): Stride for the transposed convolution.
            padding (int): Padding for the transposed convolution.
        """
        super(Model, self).__init__()
        # LazyConvTranspose1d will infer in_channels when the first forward is run
        self.deconv = nn.LazyConvTranspose1d(
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )
        # Nonlinearities
        self.hardtanh = nn.Hardtanh(min_val=-0.7, max_val=0.7)
        self.hardsigmoid = nn.Hardsigmoid()
        # Learnable per-channel scale applied after temporal pooling
        self.scale = nn.Parameter(torch.ones(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_channels, length)

        Returns:
            torch.Tensor: Output tensor of shape (batch, out_channels)
        """
        # 1) Upsample / transpose convolution: (B, outC, L_out)
        y = self.deconv(x)

        # 2) Apply a bounded nonlinearity to stabilize values
        y = self.hardtanh(y)

        # 3) Compute channel-wise global average (over time dimension)
        #    Resulting shape: (B, outC, 1)
        channel_avg = torch.mean(y, dim=2, keepdim=True)

        # 4) Produce a gate per channel in [0,1] via Hardsigmoid
        gate = self.hardsigmoid(channel_avg)

        # 5) Modulate the activations with the gate (broadcast along time dim)
        gated = y * gate

        # 6) Pool over time (sum) to produce a per-channel representation
        pooled = torch.sum(gated, dim=2)  # shape: (B, outC)

        # 7) Apply learnable per-channel scaling
        output = pooled * self.scale  # broadcasting over batch dimension

        return output

# Configuration / default sizes
batch_size = 8
in_channels = 3
input_length = 64

out_channels = 16
kernel_size = 5
stride = 2
padding = 1

def get_inputs() -> List[torch.Tensor]:
    """
    Returns example input tensors for the model. The LazyConvTranspose1d will infer
    its in_channels from the provided input tensor's shape.
    """
    x = torch.randn(batch_size, in_channels, input_length)
    return [x]

def get_init_inputs() -> List[int]:
    """
    Returns the initialization parameters required to construct the Model.
    """
    return [out_channels, kernel_size, stride, padding]