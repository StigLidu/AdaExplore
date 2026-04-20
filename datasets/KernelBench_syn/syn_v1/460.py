import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class Model(nn.Module):
    """
    Complex 1D processing module that demonstrates a pipeline combining:
      - Constant padding (nn.ConstantPad1d)
      - Max pooling with indices (nn.MaxPool1d)
      - Max unpooling (nn.MaxUnpool1d)
      - Sigmoid gating (nn.Sigmoid)
    The model pads the input, pools to obtain indices, unpools back to the padded
    resolution, constructs an attention/gating map from the unpooled signals, and
    then produces a gated residual combination.
    """
    def __init__(
        self,
        kernel_size: int,
        stride: int,
        pad_left: int,
        pad_right: int,
        pad_value: float = 0.0
    ):
        """
        Initialize the module.

        Args:
            kernel_size (int): Kernel size for max-pooling / unpooling.
            stride (int): Stride for max-pooling / unpooling.
            pad_left (int): Amount of constant padding to add to the left.
            pad_right (int): Amount of constant padding to add to the right.
            pad_value (float, optional): Constant value for padding. Defaults to 0.0.
        """
        super(Model, self).__init__()
        # Constant pad on both sides
        self.pad = nn.ConstantPad1d((pad_left, pad_right), pad_value)
        # MaxPool1d to obtain values and indices (we use return_indices=True)
        self.pool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, return_indices=True)
        # MaxUnpool1d to reconstruct using indices
        self.unpool = nn.MaxUnpool1d(kernel_size=kernel_size, stride=stride)
        # Sigmoid to form gating/attention maps
        self.sigmoid = nn.Sigmoid()

        # Save configuration for potential debugging or reconstructing output sizes
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad_left = pad_left
        self.pad_right = pad_right
        self.pad_value = pad_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
          1. Apply constant padding to the input (B, C, L) -> (B, C, L_padded)
          2. Max-pool the padded tensor to obtain pooled values and indices
          3. Unpool back to the padded size using the pooled values and indices
          4. Create a channel-aggregated attention map from the unpooled tensor
             using Sigmoid applied to the channel-sum
          5. Gate the padded input with the attention map, produce a residual,
             and combine with the unpooled structure followed by another Sigmoid.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, L)

        Returns:
            torch.Tensor: Output tensor of shape (B, C, L_padded)
        """
        # 1) Pad input
        padded = self.pad(x)  # (B, C, L_padded)

        # 2) Max-pool (capture indices)
        pooled, indices = self.pool(padded)  # pooled: (B, C, L_pool), indices same shape as pooled

        # 3) Unpool back to the padded size (using output_size to ensure exact shape)
        unpooled = self.unpool(pooled, indices, output_size=padded.size())  # (B, C, L_padded)

        # 4) Build an attention/gating map from the unpooled activations
        #    Sum across channels to aggregate, then apply Sigmoid to get [0,1] gating
        #    Result shape: (B, 1, L_padded) -> broadcastable to (B, C, L_padded)
        attention = self.sigmoid(unpooled.sum(dim=1, keepdim=True))

        # 5) Gate the padded input with attention (element-wise), create a residual
        gated = padded * attention  # (B, C, L_padded)
        residual = padded - gated   # (B, C, L_padded)

        # 6) Fuse the structures: combine unpooled structural signal with gated input,
        #    then apply a final Sigmoid nonlinearity for a bounded output.
        fused = (unpooled * 0.5) + gated  # weighted combination
        output = self.sigmoid(residual) * fused

        return output


# Configuration variables (module-level)
batch_size = 8
channels = 4
length = 128

# Pooling / padding configuration
kernel_size = 3
stride = 2
pad_left = 2
pad_right = 1
pad_value = -0.1  # use a small negative constant padding to illustrate non-zero pad

def get_inputs() -> List[torch.Tensor]:
    """
    Create and return the primary input tensor(s) for the model.

    Returns:
        List[torch.Tensor]: A single-element list containing an input tensor of shape (B, C, L).
    """
    # Random input with a mix of positive/negative values
    x = torch.randn(batch_size, channels, length)
    return [x]

def get_init_inputs() -> List:
    """
    Provide initialization parameters for the Model constructor.

    Returns:
        List: [kernel_size, stride, pad_left, pad_right, pad_value]
    """
    return [kernel_size, stride, pad_left, pad_right, pad_value]