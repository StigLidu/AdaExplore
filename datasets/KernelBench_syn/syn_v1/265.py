import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A moderately complex vision-head style module that:
    - Applies Group Normalization over the input channels
    - Runs a convolutional feature transform
    - Uses Hardswish non-linearity
    - Performs global average pooling
    - Applies RMS Normalization on the pooled features
    - Projects to output logits and adds a skip projection from the pooled input

    The module demonstrates combining normalization layers (GroupNorm, RMSNorm),
    an activation (Hardswish), and common tensor operations (conv, pooling, linear),
    forming a compact but non-trivial computation graph.
    """
    def __init__(
        self,
        in_channels: int,
        num_groups: int,
        hidden_channels: int,
        out_features: int,
        kernel_size: int = 3,
        eps: float = 1e-5
    ):
        """
        Initializes the module.

        Args:
            in_channels (int): Number of input channels.
            num_groups (int): Number of groups for GroupNorm (must divide in_channels).
            hidden_channels (int): Number of channels produced by the convolution.
            out_features (int): Size of the final output feature vector (e.g., num classes).
            kernel_size (int, optional): Convolution kernel size. Defaults to 3.
            eps (float, optional): Epsilon used for normalizations (GroupNorm and RMSNorm). Defaults to 1e-5.
        """
        super(Model, self).__init__()
        # Group normalization over channels
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=eps)
        # A small convolutional transform
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size, padding=padding, bias=True)
        # Non-linear activation
        self.act = nn.Hardswish()
        # RMS normalization over the pooled hidden dimension
        self.rmsnorm = nn.RMSNorm(hidden_channels, eps=eps)
        # Final projection from hidden representation to outputs
        self.fc = nn.Linear(hidden_channels, out_features)
        # Skip projection from pooled input channels to outputs to form a residual-like addition
        self.skip_fc = nn.Linear(in_channels, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        # Normalize input across groups
        x_norm = self.groupnorm(x)                     # (B, C, H, W)
        # Convolutional transform
        x_conv = self.conv(x_norm)                     # (B, hidden, H, W)
        # Non-linearity
        x_act = self.act(x_conv)                       # (B, hidden, H, W)
        # Global average pooling over spatial dims -> (B, hidden)
        pooled = x_act.mean(dim=(2, 3))
        # RMS normalization on the pooled vector
        pooled_norm = self.rmsnorm(pooled)             # (B, hidden)
        # Main projection
        out_main = self.fc(pooled_norm)                # (B, out_features)
        # Skip path: pool original input and project
        pooled_orig = x.mean(dim=(2, 3))               # (B, in_channels)
        out_skip = self.skip_fc(pooled_orig)           # (B, out_features)
        # Combine main path and skip path
        return out_main + out_skip

# Configuration variables
batch_size = 8
in_channels = 48        # Must be divisible by num_groups
num_groups = 8
height = 64
width = 64
hidden_channels = 128
out_features = 100
kernel_size = 3
eps = 1e-5

def get_inputs():
    """
    Returns a list containing a single input tensor suitable for the Model's forward method.
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters for the Model constructor in the same order:
    in_channels, num_groups, hidden_channels, out_features, kernel_size, eps
    """
    return [in_channels, num_groups, hidden_channels, out_features, kernel_size, eps]