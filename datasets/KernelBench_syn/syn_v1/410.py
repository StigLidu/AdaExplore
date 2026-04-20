import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex 1D processing module that:
    - Upsamples input with a ConvTranspose1d
    - Applies ReLU activation and AlphaDropout for regularization
    - Reduces temporal resolution with AdaptiveMaxPool1d
    - Adds a residual connection from a projected & pooled input
    - Finishes with a second ConvTranspose1d and tanh activation

    This demonstrates combining ConvTranspose1d, AlphaDropout, and AdaptiveMaxPool1d
    in a non-trivial computation graph with residual merging.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        dropout_p: float = 0.1,
        adaptive_output_size: int = 160,
    ):
        super(Model, self).__init__()

        # Compute a reasonable padding for ConvTranspose1d to produce (in_len * stride)
        # For typical choices like kernel_size=4, stride=2 this yields padding=1
        padding = max((kernel_size - stride) // 2, 0)

        # First transposed convolution to upsample in time and expand channels
        self.upconv = nn.ConvTranspose1d(
            in_channels, hidden_channels, kernel_size=kernel_size, stride=stride, padding=padding
        )

        # Residual projection from input channels to hidden_channels (1x1 conv)
        self.res_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)

        # Regularization and pooling layers from the provided list
        self.alpha_drop = nn.AlphaDropout(dropout_p)
        self.adaptive_pool = nn.AdaptiveMaxPool1d(adaptive_output_size)

        # Final transposed convolution to produce desired output channels (keeps length)
        # Use kernel_size=3 with padding=1 to preserve length after this layer
        self.final_upconv = nn.ConvTranspose1d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # Non-linearities
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
            x: (batch, in_channels, length)

        Returns:
            Tensor of shape (batch, out_channels, pooled_length_after_ops)
        """
        # Upsample and expand channels
        y = self.upconv(x)            # -> (batch, hidden_channels, length * stride)
        y = self.relu(y)
        y = self.alpha_drop(y)

        # Pool to a fixed temporal size
        y = self.adaptive_pool(y)     # -> (batch, hidden_channels, adaptive_output_size)

        # Prepare residual: project input channels to hidden and pool to same temporal size
        res = self.res_proj(x)        # -> (batch, hidden_channels, length)
        res = self.adaptive_pool(res) # -> (batch, hidden_channels, adaptive_output_size)

        # Merge residual
        y = y + res

        # Final up-convolution to produce outputs and squeeze values into (-1, 1)
        y = self.final_upconv(y)     # -> (batch, out_channels, adaptive_output_size)
        y = self.tanh(y)
        return y

# Configuration / default parameters for test inputs
BATCH = 8
IN_CHANNELS = 3
HIDDEN_CHANNELS = 64
OUT_CHANNELS = 2
IN_LENGTH = 128
KERNEL_SIZE = 4
STRIDE = 2
DROPOUT_P = 0.1
ADAPTIVE_OUTPUT = 160

def get_inputs():
    """
    Returns a list with one input tensor shaped (BATCH, IN_CHANNELS, IN_LENGTH).
    """
    x = torch.randn(BATCH, IN_CHANNELS, IN_LENGTH)
    return [x]

def get_init_inputs():
    """
    Returns initialization arguments for Model:
      [in_channels, hidden_channels, out_channels, kernel_size, stride, dropout_p, adaptive_output_size]
    """
    return [IN_CHANNELS, HIDDEN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, STRIDE, DROPOUT_P, ADAPTIVE_OUTPUT]