import torch
import torch.nn as nn
from typing import List

class Model(nn.Module):
    """
    Complex 3D feature gating module.

    This model accepts a 5D tensor (N, C, D, H, W), applies replication padding,
    computes global average and max pooled channel descriptors from the padded tensor,
    combines them into a gating vector which is passed through a learned linear
    projection and a Sigmoid to produce a channel-wise gate. The gate is applied
    to the original (unpadded) input and then a Threshold nonlinearity is used.
    Finally, spatial dimensions are summed to produce an (N, C) output.

    This pattern demonstrates the use of nn.ReplicationPad3d, nn.Sigmoid, and
    nn.Threshold combined with tensor reductions and a small learnable projection.
    """
    def __init__(self, channels: int, pad_size: int = 1, threshold: float = 0.0):
        """
        Initializes the module.

        Args:
            channels (int): Number of channels (C) in the input tensor.
            pad_size (int): Symmetric padding size to apply on each spatial axis.
            threshold (float): Threshold value for nn.Threshold; values below this will be set to 0.0.
        """
        super(Model, self).__init__()
        self.channels = channels
        # ReplicationPad3d expects 6 values: (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
        pad_tuple = (pad_size, pad_size, pad_size, pad_size, pad_size, pad_size)
        self.pad = nn.ReplicationPad3d(pad_tuple)

        # Learnable linear projection for channel gating: maps (C) -> (C)
        # We'll store it as a parameter matrix and bias for explicit matmul.
        self.W = nn.Parameter(torch.randn(channels, channels) * (1.0 / channels**0.5))
        self.bias = nn.Parameter(torch.zeros(channels))

        # Non-linearities
        self.sigmoid = nn.Sigmoid()
        # If input element <= threshold, set it to 0.0
        self.threshold = nn.Threshold(threshold, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Steps:
        1. Apply replication padding to x -> x_padded
        2. Compute channel-wise global average and max over spatial dims of x_padded
        3. Combine descriptors and apply a learned linear map + bias
        4. Pass through Sigmoid to create channel-wise gates
        5. Apply gates to the original (unpadded) input via broadcasting
        6. Apply Threshold nonlinearity
        7. Sum over spatial dimensions to produce an (N, C) tensor

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, D, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (N, C)
        """
        # 1) Pad
        x_padded = self.pad(x)

        # 2) Channel descriptors from padded input
        # Compute mean and max over spatial dims (D, H, W) -> shapes (N, C)
        avg_desc = x_padded.mean(dim=(2, 3, 4))
        max_desc = x_padded.amax(dim=(2, 3, 4))

        # 3) Combine descriptors
        combined = avg_desc + max_desc  # (N, C)

        # Linear projection: (N, C) @ (C, C) -> (N, C), add bias
        gating_logits = torch.matmul(combined, self.W) + self.bias  # (N, C)

        # 4) Sigmoid gating
        gates = self.sigmoid(gating_logits).view(x.shape[0], self.channels, 1, 1, 1)  # (N, C, 1, 1, 1)

        # 5) Apply gates to original input (broadcast across spatial dims)
        gated = x * gates  # (N, C, D, H, W)

        # 6) Threshold nonlinearity
        thresholded = self.threshold(gated)  # (N, C, D, H, W)

        # 7) Reduce spatial dims -> (N, C)
        out = thresholded.sum(dim=(2, 3, 4))

        return out

# Configuration / default sizes
batch_size = 8
channels = 32
depth = 10
height = 20
width = 24
pad_size = 2
threshold_value = 0.05

def get_inputs() -> List[torch.Tensor]:
    """
    Creates example input tensors for the model.

    Returns:
        A list containing a single 5D tensor of shape (N, C, D, H, W).
    """
    x = torch.randn(batch_size, channels, depth, height, width)
    return [x]

def get_init_inputs() -> List:
    """
    Returns the initialization parameters for the Model constructor in order.

    Returns:
        A list: [channels, pad_size, threshold_value]
    """
    return [channels, pad_size, threshold_value]