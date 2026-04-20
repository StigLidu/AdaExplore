import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

class Model(nn.Module):
    """
    A composite 3D reconstruction and channel-gating module.

    The model expects as input:
      - pooled: result of a MaxPool3d operation (Tensor of shape [B, C, D', H', W'])
      - indices: pooling indices returned by F.max_pool3d with return_indices=True
      - output_size: the original size tuple for unpooling (e.g., x.size())

    Pipeline:
      1. MaxUnpool3d to approximate the original 3D spatial resolution.
      2. SiLU activation applied element-wise.
      3. Channel-wise global average pooling across (D, H, W) producing a channel descriptor.
      4. LogSigmoid applied to the descriptor to yield gating coefficients.
      5. Channel gating: scale the unpooled tensor by the descriptor (broadcasted over spatial dims).
    """
    def __init__(self, kernel_size: int, stride: int, padding: int = 0):
        """
        Args:
            kernel_size (int): Kernel size used for unpooling (must match the pool kernel).
            stride (int): Stride used for unpooling (must match the pool stride).
            padding (int, optional): Padding for unpooling (defaults to 0).
        """
        super(Model, self).__init__()
        # MaxUnpool3d to invert MaxPool3d partially
        self.unpool = nn.MaxUnpool3d(kernel_size=kernel_size, stride=stride, padding=padding)
        # Non-linearities for activation and gating
        self.silu = nn.SiLU()
        self.logsigmoid = nn.LogSigmoid()

    def forward(self, pooled: torch.Tensor, indices: torch.Tensor, output_size: Tuple[int, ...]) -> torch.Tensor:
        """
        Args:
            pooled (torch.Tensor): Pooled input of shape (B, C, D', H', W').
            indices (torch.Tensor): Indices from max_pool3d corresponding to pooled.
            output_size (tuple): The desired output size for unpooling (original tensor size).

        Returns:
            torch.Tensor: Reconstructed and channel-gated tensor of shape output_size.
        """
        # 1) Unpool back to a higher-resolution 3D grid (approximate inverse of pooling)
        unpooled = self.unpool(pooled, indices, output_size=output_size)

        # 2) Apply element-wise SiLU non-linearity
        activated = self.silu(unpooled)

        # 3) Channel-wise descriptor: global average over spatial dimensions (D, H, W)
        # Resulting shape: (B, C, 1, 1, 1)
        channel_descriptor = activated.mean(dim=(2, 3, 4), keepdim=True)

        # 4) Convert descriptor to gating coefficients using LogSigmoid (stabilized gating in log-space)
        gating = self.logsigmoid(channel_descriptor)  # same shape as channel_descriptor

        # 5) Apply channel-wise gating (broadcast over spatial dims)
        output = unpooled * gating

        return output

# Configuration variables
batch_size = 4
channels = 6
depth = 8    # must be divisible by kernel_size (below)
height = 12  # must be divisible by kernel_size
width = 12   # must be divisible by kernel_size

kernel_size = 2
stride = 2
padding = 0

def get_inputs() -> List:
    """
    Prepares inputs for the model by:
      - Creating a random 5D tensor: (B, C, D, H, W)
      - Applying max_pool3d with return_indices=True to obtain pooled and indices
      - Returning [pooled, indices, original_size] suitable for the Model.forward

    Returns:
        list: [pooled_tensor, indices_tensor, original_size_tuple]
    """
    x = torch.randn(batch_size, channels, depth, height, width)
    # Perform max_pool3d to obtain pooled representation and pooling indices
    pooled, indices = F.max_pool3d(x, kernel_size=kernel_size, stride=stride, padding=padding, return_indices=True)
    # Provide the original size for unpooling
    original_size = x.size()
    return [pooled, indices, original_size]

def get_init_inputs() -> List:
    """
    Returns initialization parameters for Model: kernel_size, stride, padding
    """
    return [kernel_size, stride, padding]