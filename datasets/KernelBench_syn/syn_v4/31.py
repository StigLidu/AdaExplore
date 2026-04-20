import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class Model(nn.Module):
    """
    Complex 1D processing module that demonstrates a small pipeline:
    1) Constant padding on both sides of the temporal dimension
    2) 1D convolution to mix channel and local temporal information
    3) Channel-wise dropout (Dropout1d)
    4) HardTanh non-linearity for bounded activations
    5) Adaptive average pooling to a fixed output temporal size

    The network returns a flattened feature tensor of shape (batch_size, mid_channels * pool_output_size).
    """
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        kernel_size: int,
        pad_left: int,
        pad_right: int,
        pad_value: float,
        dropout_prob: float,
        tanh_min: float,
        tanh_max: float,
        pool_output_size: int
    ):
        """
        Initializes the processing pipeline.

        Args:
            in_channels (int): Number of input channels.
            mid_channels (int): Number of channels produced by the convolution.
            kernel_size (int): Kernel size for the Conv1d layer.
            pad_left (int): Amount of constant padding to add on the left.
            pad_right (int): Amount of constant padding to add on the right.
            pad_value (float): Constant value to use for padding.
            dropout_prob (float): Dropout1d probability.
            tanh_min (float): Minimum value for HardTanh.
            tanh_max (float): Maximum value for HardTanh.
            pool_output_size (int): Output temporal size for AdaptiveAvgPool1d.
        """
        super(Model, self).__init__()
        # Padding layer (left, right)
        self.pad = nn.ConstantPad1d((pad_left, pad_right), pad_value)
        # 1D convolution: no extra padding because we already padded explicitly
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel_size, stride=1, padding=0, bias=True)
        # Dropout across channels
        self.dropout = nn.Dropout1d(p=dropout_prob)
        # HardTanh activation for bounded activations
        self.hardtanh = nn.Hardtanh(min_val=tanh_min, max_val=tanh_max)
        # Reduce/reshape temporal dimension to a fixed size
        self.pool = nn.AdaptiveAvgPool1d(output_size=pool_output_size)

        # Store some shapes for potential debugging/inspection
        self._out_feature_size = mid_channels * pool_output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, seq_len).

        Returns:
            torch.Tensor: Flattened feature tensor of shape (batch_size, mid_channels * pool_output_size).
        """
        # 1) Constant padding in temporal dimension
        x = self.pad(x)
        # 2) Convolution to mix channels and local temporal context
        x = self.conv(x)
        # 3) Channel-wise dropout
        x = self.dropout(x)
        # 4) Bounded non-linearity
        x = self.hardtanh(x)
        # 5) Pool to fixed temporal length
        x = self.pool(x)
        # Flatten per-example
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return x

# Module-level configuration variables
batch_size = 8
in_channels = 16
mid_channels = 32
seq_len = 128
kernel_size = 5
pad_left = 2
pad_right = 3
pad_value = 0.1
dropout_prob = 0.25
tanh_min = -0.8
tanh_max = 0.8
pool_output_size = 16

def get_inputs() -> List[torch.Tensor]:
    """
    Produces example input tensors for the model.

    Returns:
        A list containing a single input tensor of shape (batch_size, in_channels, seq_len).
    """
    x = torch.randn(batch_size, in_channels, seq_len)
    return [x]

def get_init_inputs() -> List:
    """
    Returns initialization parameters for the Model constructor in the same order
    as the __init__ signature.
    """
    return [
        in_channels,
        mid_channels,
        kernel_size,
        pad_left,
        pad_right,
        pad_value,
        dropout_prob,
        tanh_min,
        tanh_max,
        pool_output_size
    ]