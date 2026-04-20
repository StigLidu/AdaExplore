import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Any

class Model(nn.Module):
    """
    Complex model combining 3D replication padding, RMS-like normalization,
    1D constant padding, ReLU6 activation, and a learnable channel scaling.
    
    Pipeline:
    1. ReplicationPad3d to pad spatial dimensions.
    2. Compute RMS across channels and normalize the tensor.
    3. Collapse spatial dims into a single length and apply ConstantPad1d.
    4. Apply ReLU6 activation.
    5. Apply a per-channel learnable scaling.
    6. Reduce the final length dimension to produce (batch, channels) output.
    """
    def __init__(
        self,
        channels: int,
        pad3d: Tuple[int, int, int, int, int, int],
        const_pad: Tuple[int, int],
        const_value: float = 0.0,
        eps: float = 1e-5,
    ):
        """
        Args:
            channels (int): Number of channels in the input tensor.
            pad3d (tuple): 6-int tuple for ReplicationPad3d: (left, right, top, bottom, front, back).
            const_pad (tuple): 2-int tuple for ConstantPad1d: (pad_left, pad_right).
            const_value (float): Constant value for ConstantPad1d.
            eps (float): Small epsilon for numerical stability in RMS normalization.
        """
        super(Model, self).__init__()
        # Layers
        self.rep_pad3d = nn.ReplicationPad3d(pad3d)
        self.const_pad1d = nn.ConstantPad1d(const_pad, const_value)
        self.relu6 = nn.ReLU6()
        # Parameters and hyperparams
        self.channels = channels
        self.eps = eps
        # Learnable per-channel scale (broadcast over length dimension later)
        self.register_parameter("scale", nn.Parameter(torch.ones(1, channels, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Input:
            x: Tensor of shape (batch, channels, depth, height, width)

        Returns:
            Tensor of shape (batch, channels) after spatial collapse, padding, activation and scaling.
        """
        # 1) Replication padding on 3D spatial dims -> (N, C, Dp, Hp, Wp)
        x = self.rep_pad3d(x)

        # 2) Compute RMS across channels for each spatial location: shape (N, 1, Dp, Hp, Wp)
        #    Then normalize channels by this RMS (broadcastable)
        rms = torch.sqrt(torch.mean(x * x, dim=1, keepdim=True) + self.eps)
        x = x / rms

        # 3) Collapse spatial dims into a single length for 1D padding:
        #    (N, C, Dp * Hp * Wp)
        N, C = x.shape[0], x.shape[1]
        x = x.contiguous().view(N, C, -1)

        # 4) ConstantPad1d on the collapsed length dimension
        x = self.const_pad1d(x)

        # 5) Non-linearity: ReLU6
        x = self.relu6(x)

        # 6) Apply learnable per-channel scaling (broadcast over length)
        x = x * self.scale

        # 7) Reduce over the length dimension to produce a compact (N, C) representation
        out = x.sum(dim=2)

        return out

# Module-level configuration (defaults for get_inputs / get_init_inputs)
batch_size = 2
channels = 8
depth = 6
height = 5
width = 7
# ReplicationPad3d expects (left, right, top, bottom, front, back)
pad3d = (1, 2, 0, 1, 1, 0)
# ConstantPad1d expects (pad_left, pad_right)
const_pad = (3, 2)
const_value = -0.25
eps = 1e-6

def get_inputs() -> List[torch.Tensor]:
    """
    Returns a list with one input tensor for the forward pass.
    Shape: (batch_size, channels, depth, height, width)
    """
    x = torch.randn(batch_size, channels, depth, height, width)
    return [x]

def get_init_inputs() -> List[Any]:
    """
    Returns initialization parameters for the Model: [channels, pad3d, const_pad, const_value, eps]
    """
    return [channels, pad3d, const_pad, const_value, eps]