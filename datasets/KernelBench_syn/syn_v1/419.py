import torch
import torch.nn as nn

# Configuration / shapes
BATCH = 4
IN_CHANNELS = 3
INPUT_LENGTH = 64

PAD = 2  # ZeroPad1d padding on both sides
OUT_CHANNELS = 16
KERNEL_SIZE = 5
STRIDE = 2

EMBED_DIM = 32  # final per-timestep embedding dimension after linear projection

LRN_SIZE = 5
LRN_ALPHA = 1e-4
LRN_BETA = 0.75
LRN_K = 2.0

class Model(nn.Module):
    """
    A moderately complex 1D signal processing module that:
      1. Zero-pads the input signal.
      2. Applies a transposed convolution (LazyConvTranspose1d) to expand temporal resolution.
      3. Applies a non-linearity (GELU).
      4. Applies Local Response Normalization across channels.
      5. Projects channel features to a per-timestep embedding via a linear layer.
      6. L2-normalizes the final per-timestep embeddings.

    Notes:
      - LazyConvTranspose1d allows the module to be instantiated without specifying in_channels;
        it will be inferred at first forward pass.
      - Input expected shape: (batch, in_channels, length)
      - Output shape: (batch, out_length, embed_dim) -- time dimension moved to axis 1
    """
    def __init__(self):
        super(Model, self).__init__()
        # Pad both sides of the last dimension by PAD
        self.pad = nn.ZeroPad1d(PAD)

        # Lazy transpose convolution will infer in_channels on first forward
        # We set out_channels, kernel_size and stride explicitly
        self.deconv = nn.LazyConvTranspose1d(
            OUT_CHANNELS,
            kernel_size=KERNEL_SIZE,
            stride=STRIDE,
            bias=True
        )

        # Local response normalization across channels
        self.lrn = nn.LocalResponseNorm(size=LRN_SIZE, alpha=LRN_ALPHA, beta=LRN_BETA, k=LRN_K)

        # Small non-linearity
        self.act = nn.GELU()

        # Linear projection applied per temporal position: channels -> EMBED_DIM
        # We'll instantiate as nn.Linear which expects (batch*length, channels) or we can use nn.Conv1d with kernel_size=1.
        self.proj = nn.Linear(OUT_CHANNELS, EMBED_DIM, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch, in_channels, length)

        Returns:
            Tensor of shape (batch, out_length, EMBED_DIM), L2-normalized along the embedding dimension.
        """
        # 1) Zero pad time dimension: ZeroPad1d expects (N, C, L) and pads the last dim
        x = self.pad(x)

        # 2) Transposed convolution to upsample temporal resolution and mix channels
        x = self.deconv(x)  # -> (batch, OUT_CHANNELS, out_length)

        # 3) Non-linearity
        x = self.act(x)

        # 4) Local Response Normalization across channels
        x = self.lrn(x)

        # 5) Project channels -> EMBED_DIM per time step
        # Convert to (batch, length, channels) for linear which acts on last dim
        x = x.permute(0, 2, 1)  # (batch, out_length, OUT_CHANNELS)
        # Apply linear projection across the channel dimension
        # nn.Linear is applied to last dim automatically if x is (..., in_features)
        x = self.proj(x)  # (batch, out_length, EMBED_DIM)

        # 6) L2 normalize embeddings along the embed dim (for each time step)
        x = x / (torch.norm(x, p=2, dim=2, keepdim=True) + 1e-8)

        return x

def get_inputs():
    """
    Returns a list of input tensors to be passed to Model.forward.
    Input shape: (BATCH, IN_CHANNELS, INPUT_LENGTH)
    """
    x = torch.randn(BATCH, IN_CHANNELS, INPUT_LENGTH)
    return [x]

def get_init_inputs():
    """
    Returns any initialization parameters required by the module.
    LazyConvTranspose1d is used, so no explicit in_channels needed here.
    """
    return []