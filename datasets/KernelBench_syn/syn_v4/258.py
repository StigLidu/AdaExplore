import torch
import torch.nn as nn

# Configuration
BATCH_SIZE = 32
IN_CHANNELS = 64   # Will be inferred by LazyConv1d, but used for generating inputs
SEQ_LEN = 512
HIDDEN_CHANNELS = 256  # Internal hidden channel size used by the module

class Model(nn.Module):
    """
    A convolutional gated block that demonstrates lazy initialization and gated activations.
    Pipeline:
      1. Lazy 1D convolution producing 2 * hidden_channels channels
      2. GLU along the channel dimension to produce hidden_channels channels
      3. Randomized leaky ReLU (RReLU) activation
      4. Lazy 1D convolution on the original input to form a channel-matched shortcut
      5. Residual addition
      6. Lazy 1D pointwise projection
      7. Global average pooling over the temporal dimension -> (batch, hidden_channels)
    This structure exercises LazyConv1d, GLU, and RReLU in a compact, realistic block.
    """
    def __init__(self, hidden_channels: int = HIDDEN_CHANNELS):
        super(Model, self).__init__()
        self.hidden_channels = hidden_channels

        # Produces 2 * hidden_channels so GLU can split it into values and gates
        self.conv = nn.LazyConv1d(out_channels=self.hidden_channels * 2, kernel_size=3, padding=1)

        # Shortcut to map input channels to hidden_channels (lazy to accept any incoming in_channels)
        self.shortcut = nn.LazyConv1d(out_channels=self.hidden_channels, kernel_size=1)

        # Pointwise projection after residual addition
        self.proj = nn.LazyConv1d(out_channels=self.hidden_channels, kernel_size=1)

        # Gated Linear Unit reduces channels from 2*H -> H
        self.glu = nn.GLU(dim=1)

        # Randomized leaky ReLU for non-linearity with train/test behavior
        self.rrelu = nn.RReLU(lower=0.125, upper=0.333, inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, seq_len)

        Returns:
            Tensor of shape (batch_size, hidden_channels) after global average pooling
        """
        # Save residual (original input)
        residual = x

        # 1) Convolution -> (B, 2*H, L)
        x = self.conv(x)

        # 2) GLU -> (B, H, L)
        x = self.glu(x)

        # 3) RReLU activation -> (B, H, L)
        x = self.rrelu(x)

        # 4) Shortcut mapping from original input to (B, H, L)
        sc = self.shortcut(residual)

        # 5) Residual addition
        x = x + sc

        # 6) Pointwise projection to refine features -> (B, H, L)
        x = self.proj(x)

        # 7) Global average pooling over sequence length -> (B, H)
        out = x.mean(dim=2)

        return out

def get_inputs():
    """
    Returns:
        list containing a single input tensor shaped (BATCH_SIZE, IN_CHANNELS, SEQ_LEN)
    """
    A = torch.randn(BATCH_SIZE, IN_CHANNELS, SEQ_LEN)
    return [A]

def get_init_inputs():
    """
    Returns initialization inputs for constructing the Model (hidden_channels).
    The Model __init__ accepts this optional parameter.
    """
    return [HIDDEN_CHANNELS]