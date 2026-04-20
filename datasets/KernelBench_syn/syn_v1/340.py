import torch
import torch.nn as nn

class Model(nn.Module):
    """
    3D feature gating module combining BatchNorm3d, channel-wise squeeze-and-excitation
    style gating (implemented with Linear layers), SELU nonlinearity, and LogSigmoid-based
    numerical gating. The module normalizes spatial features, squeezes global context,
    passes through a small bottleneck MLP, computes a sigmoid gate using LogSigmoid
    (for numerical stability demonstration), and applies the gate to the original features
    while also adding a SELU-activated normalized path (residual-style combination).
    """
    def __init__(self, channels: int, reduction: int = 4):
        """
        Initializes the gating module.

        Args:
            channels (int): Number of input channels (C dimension).
            reduction (int, optional): Bottleneck reduction factor for the channel MLP.
                                       Must be >= 1. Defaults to 4.
        """
        super(Model, self).__init__()
        if reduction < 1:
            raise ValueError("reduction must be >= 1")
        hidden = max(1, channels // reduction)

        # Normalize spatial features per channel
        self.bn = nn.BatchNorm3d(channels)

        # Bottleneck MLP to compute channel-wise gates (squeeze + excite style)
        self.reduce = nn.Linear(channels, hidden, bias=True)
        self.expand = nn.Linear(hidden, channels, bias=True)

        # Nonlinearities
        self.selu = nn.SELU()
        self.logsigmoid = nn.LogSigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, D, H, W).

        Returns:
            torch.Tensor: Output tensor of same shape (B, C, D, H, W) after gating and
                          residual combination.
        """
        # Keep a residual copy of the original input
        residual = x

        # 1) Normalize across batch and spatial dims per channel
        normalized = self.bn(x)

        # 2) Squeeze: global spatial average to get channel descriptors (B, C)
        # Using mean across D, H, W dims
        s = normalized.mean(dim=(2, 3, 4))  # shape: (B, C)

        # 3) Bottleneck MLP: reduce -> nonlinearity -> expand
        s_reduced = self.reduce(s)          # (B, hidden)
        s_activated = self.selu(s_reduced)  # (B, hidden)
        s_expanded = self.expand(s_activated)  # (B, C)

        # 4) Compute gate using LogSigmoid for numerical stability; convert back to sigmoid
        # LogSigmoid returns log(sigmoid(x)), so exp(...) yields sigmoid(x)
        gate = torch.exp(self.logsigmoid(s_expanded))  # (B, C), values in (0,1)

        # 5) Reshape gate and apply to the original residual features
        gate = gate.view(gate.size(0), gate.size(1), 1, 1, 1)  # (B, C, 1, 1, 1)
        gated = residual * gate  # gated original features

        # 6) Combine gated original with SELU-activated normalized path (residual-style)
        out = gated + self.selu(normalized)

        return out

# Configuration variables
batch_size = 8
channels = 32
depth = 16
height = 32
width = 32
reduction = 4

def get_inputs():
    """
    Generates a random input tensor compatible with the model.

    Returns:
        list: Single-element list containing the input tensor x.
    """
    x = torch.randn(batch_size, channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for the Model constructor:
    channels and reduction factor.

    Returns:
        list: [channels, reduction]
    """
    return [channels, reduction]