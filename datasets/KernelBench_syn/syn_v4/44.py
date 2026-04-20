import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex vision-inspired module that combines convolutional processing,
    synchronized batch normalization, and two different nonlinearities
    (HardTanh and Softsign) to produce a residual-style feature transform.

    The forward pass:
    1. Applies a 3x3 convolution to expand channels.
    2. Applies SyncBatchNorm to stabilize statistics across devices.
    3. Applies HardTanh nonlinearity (clamped activation).
    4. Computes a channel-wise descriptor via spatial global average.
    5. Applies Softsign to the descriptor and uses it to modulate features
       (like a lightweight channel attention).
    6. Projects back to the original channel count with a 1x1 convolution.
    7. Adds a residual connection and applies a final Softsign for bounded output.
    """
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        """
        Initializes the module.

        Args:
            in_channels (int): Number of channels in the input tensor.
            hidden_channels (int): Number of channels in the hidden representation.
            kernel_size (int): Kernel size for the first convolution (uses padding to preserve spatial dims).
        """
        super(Model, self).__init__()
        padding = kernel_size // 2
        # Expand spatial features
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size, padding=padding, bias=False)
        # Normalize across devices / processes (keeps feature dimension consistent)
        self.syncbn = nn.SyncBatchNorm(num_features=hidden_channels)
        # Bounded activation to clip extreme values
        self.hardtanh = nn.Hardtanh(min_val=-0.8, max_val=0.8)
        # Channel-wise modulation (applied to the descriptor)
        self.softsign = nn.Softsign()
        # Project back to original channels
        self.conv2 = nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=True)

        # Small re-scaling parameter to control attention strength; initialized near zero
        self.scale = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of the same shape as input.
        """
        # Save residual for skip connection
        residual = x

        # 1) Expand features with convolution
        y = self.conv1(x)

        # 2) Apply synchronized batch normalization
        y = self.syncbn(y)

        # 3) HardTanh nonlinearity to clamp activations
        y = self.hardtanh(y)

        # 4) Channel-wise descriptor via global average pooling (B, C, 1, 1)
        desc = torch.mean(y, dim=(2, 3), keepdim=True)

        # 5) Modulate descriptor with a learned tiny scaling then Softsign to keep values bounded
        desc = self.softsign(self.scale * desc)  # values in (-1,1) roughly

        # 6) Broadcast multiply (channel-wise gating)
        y = y * (1.0 + desc)

        # 7) Project back to original channels
        y = self.conv2(y)

        # 8) Residual connection
        out = y + residual

        # 9) Final Softsign to keep outputs smooth and bounded
        out = self.softsign(out)

        return out

# Configuration variables
batch_size = 8
in_channels = 16
hidden_channels = 48
height = 32
width = 32
kernel_size = 3

def get_inputs():
    """
    Returns a list with the primary input tensor for the model.
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters required to construct the Model.
    Order: [in_channels, hidden_channels, kernel_size]
    """
    return [in_channels, hidden_channels, kernel_size]