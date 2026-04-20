import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Complex PyTorch kernel module demonstrating:
- LazyConv1d (lazy channel initialization)
- RMSNorm for channel-wise normalization
- Softshrink activation
- Gated residual pattern with projection and element-wise modulation

Structure follows the examples:
- Model class inheriting from nn.Module
- get_inputs() to produce runtime inputs
- get_init_inputs() for init-time parameters
- Module-level configuration variables
"""

# Configuration variables
batch_size = 8
in_channels = 16   # input channels for the sequence data
out_channels = 32  # number of output channels produced by learned convolutions
seq_len = 512      # temporal/spatial length of the 1D signal
kernel_size = 5    # kernel size for the primary convolution

class Model(nn.Module):
    """
    A sequence processing block that:
    - Applies a lazily-initialized 1D convolution to the input.
    - Projects the input into the same output channel space via a lazy 1x1 conv.
    - Produces a learned gate from the main conv output and uses it to modulate features.
    - Normalizes across channels with RMSNorm (applied on the last dimension).
    - Applies Softshrink activation and a residual skip connection.

    Input shape: (batch_size, in_channels, seq_len)
    Output shape: (batch_size, out_channels, seq_len)
    """
    def __init__(self,
                 out_channels: int = out_channels,
                 kernel_size: int = kernel_size,
                 softshrink_lambda: float = 0.5):
        """
        Args:
            out_channels: Number of output channels for conv layers and RMSNorm.
            kernel_size: Kernel size for the primary convolution.
            softshrink_lambda: Lambda parameter for Softshrink activation.
        """
        super(Model, self).__init__()

        # Primary convolution - lazy in_channels initialization
        # Padding chosen to preserve seq_len
        self.conv = nn.LazyConv1d(out_channels=out_channels,
                                  kernel_size=kernel_size,
                                  padding=(kernel_size // 2))

        # 1x1 projection branch (also lazy) to match the main conv channels for residuals
        self.proj = nn.LazyConv1d(out_channels=out_channels,
                                  kernel_size=1)

        # Gate generator: produces per-channel, per-position gating weights
        # This is a small pointwise conv (in_channels will be known after lazy conv initializes)
        # We'll create it as a standard Conv1d but initialize its in_channels in forward using weight reshape if needed.
        # To keep module registration consistent, define it using known out_channels as both in/out channels.
        self.gate_conv = nn.Conv1d(in_channels=out_channels,
                                   out_channels=out_channels,
                                   kernel_size=1)

        # RMSNorm normalizes over the last dimension; because we'll permute to (N, L, C),
        # normalized_shape should be number of channels => out_channels
        self.rms = nn.RMSNorm(out_channels)

        # Softshrink non-linearity
        self.soft = nn.Softshrink(lambd=softshrink_lambda)

        # Small dropout to add regularization (optional, deterministic if eval)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation:
        1. conv_out = conv(x)                           # (N, out_channels, L)
        2. proj_out = proj(x)                           # (N, out_channels, L)
        3. gate = sigmoid(gate_conv(conv_out))         # (N, out_channels, L)
        4. modulated = conv_out * gate + proj_out
        5. permute -> (N, L, C) and apply RMSNorm over C
        6. apply Softshrink, dropout, add residual (pre-norm modulated)
        7. permute back -> (N, C, L) and return

        Args:
            x: Input tensor of shape (batch, in_channels, seq_len)

        Returns:
            Tensor of shape (batch, out_channels, seq_len)
        """
        # Primary conv branch (lazy init will infer in_channels on first call)
        conv_out = self.conv(x)            # (N, out_channels, L)

        # Projection branch (lazy init)
        proj_out = self.proj(x)            # (N, out_channels, L)

        # Gate generation, sigmoid to (0,1)
        # gate_conv expects input channels == out_channels; ensure shapes align.
        # After lazy init conv_out will have channels == gate_conv.in_channels.
        gate = torch.sigmoid(self.gate_conv(conv_out))  # (N, out_channels, L)

        # Modulate and combine with projection branch
        modulated = conv_out * gate + proj_out  # (N, out_channels, L)

        # Permute to (N, L, C) because RMSNorm normalizes over the last dimension
        mod_permuted = modulated.permute(0, 2, 1)  # (N, L, C)

        # RMS normalization across channels
        normalized = self.rms(mod_permuted)       # (N, L, C)

        # Non-linear shrinkage
        activated = self.soft(normalized)         # (N, L, C)

        # Optional regularization
        activated = self.dropout(activated)

        # Residual connection in permuted space (pre-norm mod_permuted used as residual)
        output_permuted = activated + mod_permuted

        # Permute back to (N, C, L)
        output = output_permuted.permute(0, 2, 1)  # (N, out_channels, L)

        return output

# Utility functions to match the example file patterns
def get_inputs():
    """
    Creates a representative input tensor for the model.

    Returns:
        List containing a single tensor of shape (batch_size, in_channels, seq_len)
    """
    x = torch.randn(batch_size, in_channels, seq_len)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters if any are required externally.
    For this model we don't require special initialization from outside.

    Returns:
        Empty list
    """
    return []