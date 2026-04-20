import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class Model(nn.Module):
    """
    1D sequence processing module that demonstrates a small encoder-like pattern:
    - 1D convolution expands channels to 2*hidden for GLU gating
    - GLU splits & gates along the channel dimension
    - Tanh non-linearity applied to gated output
    - MaxPool1d (with indices) reduces temporal length
    - MaxUnpool1d restores temporal length using the saved indices
    - Residual connection between pre-pooled gated features and the unpooled result
    - Final 1x1 conv reduces channels to desired output, followed by global temporal average
    
    This model uses nn.GLU, nn.Tanh, and nn.MaxUnpool1d from the provided layer list.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, conv_kernel: int = 3, pool_kernel: int = 2):
        """
        Args:
            in_channels: Number of input channels.
            hidden_channels: Number of hidden channels after gating (GLU output channels).
            out_channels: Number of output channels produced before global pooling.
            conv_kernel: Kernel size for the initial Conv1d expansion.
            pool_kernel: Kernel size / stride for MaxPool1d and MaxUnpool1d.
        """
        super(Model, self).__init__()
        # Expand channels to 2 * hidden_channels so GLU can split in half along channel dim
        self.conv_expand = nn.Conv1d(in_channels, hidden_channels * 2, kernel_size=conv_kernel, padding=(conv_kernel // 2))
        self.glu = nn.GLU(dim=1)  # split along channel dimension
        self.tanh = nn.Tanh()
        # MaxPool1d that returns indices (used later by MaxUnpool1d)
        self.pool = nn.MaxPool1d(kernel_size=pool_kernel, stride=pool_kernel, return_indices=True)
        self.unpool = nn.MaxUnpool1d(kernel_size=pool_kernel, stride=pool_kernel)
        # Final 1x1 convolution to map hidden channels to out_channels
        self.conv_project = nn.Conv1d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, in_channels, seq_len)

        Returns:
            Tensor of shape (batch, out_channels) produced by global averaging over the time dimension.
        """
        # Expand channels and apply GLU gating
        expanded = self.conv_expand(x)                        # (B, 2*H, L)
        gated = self.glu(expanded)                            # (B, H, L)
        gated = self.tanh(gated)                              # (B, H, L)

        # Save residual for later
        residual = gated.clone()                              # (B, H, L)

        # Pool down (capture indices)
        pooled, indices = self.pool(gated)                    # (B, H, L//pool), indices shape same as pooled

        # Unpool back to original temporal length using indices and explicit output_size
        # MaxUnpool1d requires passing output_size to ensure exact restoration when needed.
        output_size = gated.size()                            # (B, H, L)
        unpooled = self.unpool(pooled, indices, output_size=output_size)  # (B, H, L)

        # Combine residual (pre-pool) with unpooled signal (simple addition) and project
        combined = unpooled + residual                        # (B, H, L)
        projected = self.conv_project(combined)               # (B, out_channels, L)

        # Global temporal average to produce a compact per-channel vector
        out = projected.mean(dim=2)                           # (B, out_channels)

        return out

# Module-level configuration variables
BATCH_SIZE = 8
IN_CHANNELS = 12
HIDDEN_CHANNELS = 64
OUT_CHANNELS = 16
SEQ_LEN = 128
CONV_KERNEL = 3
POOL_KERNEL = 2

def get_inputs() -> List[torch.Tensor]:
    """
    Creates a random input tensor matching the configured shapes.

    Returns:
        List containing a single input tensor [x] of shape (BATCH_SIZE, IN_CHANNELS, SEQ_LEN)
    """
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, SEQ_LEN)
    return [x]

def get_init_inputs() -> List:
    """
    Initialization parameters for the Model constructor.

    Returns:
        List of init arguments: [in_channels, hidden_channels, out_channels, conv_kernel, pool_kernel]
    """
    return [IN_CHANNELS, HIDDEN_CHANNELS, OUT_CHANNELS, CONV_KERNEL, POOL_KERNEL]