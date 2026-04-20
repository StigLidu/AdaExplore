import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex convolutional module that demonstrates:
    - Boundary handling via replication padding
    - Two convolutional streams with different receptive fields (standard and dilated)
    - Concatenation of feature streams and channel-wise gating using global context (AdaptiveAvgPool2d)
    - Final pointwise projection with a residual connection (with optional projection)
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        pad_std: int = 1,
        pad_dil: int = 2,
        pool_output_size: tuple = (1, 1)
    ):
        """
        Initializes the Model.

        Args:
            in_channels (int): Number of channels in the input tensor.
            hidden_channels (int): Number of channels in the intermediate convolutional streams.
            out_channels (int): Number of channels in the output tensor.
            pad_std (int): Replication padding for the standard 3x3 conv (default 1 to preserve spatial size).
            pad_dil (int): Replication padding for the dilated 3x3 conv with dilation=2 (default 2 to preserve spatial size).
            pool_output_size (tuple): Output size for AdaptiveAvgPool2d (typically (1,1) for channel gating).
        """
        super(Model, self).__init__()

        # Padding layers to preserve spatial dims before convolutions
        self.pad_std = nn.ReplicationPad2d(pad_std)
        self.pad_dil = nn.ReplicationPad2d(pad_dil)

        # Two parallel convolutional streams
        # Standard 3x3 conv
        self.conv_std = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=0, bias=True)
        # Dilated 3x3 conv (effective receptive field 5x5 with dilation=2)
        self.conv_dil = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=0, dilation=2, bias=True)

        # Adaptive average pooling to extract global context for channel gating
        self.global_pool = nn.AdaptiveAvgPool2d(pool_output_size)

        # Pointwise projection after concatenation and gating
        self.project = nn.Conv2d(2 * hidden_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

        # If input channels differ from out_channels, use a projection for residual connection
        if in_channels != out_channels:
            self.res_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        else:
            self.res_proj = nn.Identity()

        # Small stabilization parameter could be useful for gating (kept as buffer for determinism)
        self.register_buffer("eps", torch.tensor(1e-6))

        # Activation
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining the parallel convolutional streams, channel-wise gating and residual projection.

        Steps:
        1. Apply replication padding and a 3x3 convolution -> activation.
        2. Apply replication padding and a dilated 3x3 convolution on the result -> activation.
        3. Concatenate the two streams along the channel dimension.
        4. Compute a global context via AdaptiveAvgPool2d and convert to gating weights using sigmoid.
        5. Apply channel-wise gating to the concatenated features.
        6. Project gated features with a 1x1 convolution.
        7. Add a residual connection (projected input if necessary).

        Args:
            x (torch.Tensor): Input tensor of shape (B, in_channels, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (B, out_channels, H, W)
        """
        # Stream A: standard conv path
        a = self.pad_std(x)
        a = self.conv_std(a)
        a = self.act(a)

        # Stream B: apply an additional convolutional step with dilation to expand receptive field
        # Use replication padding sized for dilated convolution
        b = self.pad_dil(a)
        b = self.conv_dil(b)
        b = self.act(b)

        # Concatenate streams: channels = hidden + hidden = 2*hidden
        concat = torch.cat([a, b], dim=1)

        # Global context for channel-wise gating
        context = self.global_pool(concat)  # (B, 2*hidden, 1, 1) if pool_output_size=(1,1)
        gating = torch.sigmoid(context)  # values in (0,1)

        # Apply gating (broadcast multiplication)
        gated = concat * (gating + self.eps)

        # Project to desired output channels
        out = self.project(gated)

        # Residual connection (project input if channel dims differ)
        res = self.res_proj(x)
        return out + res

# Configuration variables
batch_size = 8
in_channels = 3
hidden_channels = 64
out_channels = 16
height = 128
width = 128
pad_std = 1   # keep spatial dims for 3x3 conv
pad_dil = 2   # keep spatial dims for dilated conv (dilation=2 -> effective kernel size 5)
pool_output_size = (1, 1)

def get_inputs():
    """
    Generates a list with a single input tensor matching the configuration above.
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for constructing the Model externally if needed.
    """
    return [in_channels, hidden_channels, out_channels, pad_std, pad_dil, pool_output_size]