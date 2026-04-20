import torch
import torch.nn as nn

# Configuration
batch_size = 8
in_channels = 3
height = 64
width = 64

# LPPool parameters
lp_p = 3
lp_kernel = 3
lp_stride = 2

# Convolution expansion and dropout
expand_factor = 2
dropout_p = 0.1

# Adaptive pool output size (sequence length after 1D pooling)
adaptive_out = 8

class Model(nn.Module):
    """
    Complex model that combines LPPool2d, AlphaDropout, a 1x1 convolutional expansion,
    and AdaptiveMaxPool1d to produce a compact representation of a 4D image-like input.
    The pipeline:
      1) Lp pooling over spatial dims to reduce / aggregate neighborhoods
      2) AlphaDropout for regularization that preserves self-normalizing properties
      3) Channel expansion via 1x1 convolution + GELU activation
      4) Flatten spatial dims to a sequence and apply AdaptiveMaxPool1d
      5) L2-normalize across channel dimension for each output position
    """
    def __init__(
        self,
        in_channels: int = in_channels,
        expand_factor: int = expand_factor,
        lp_p: int = lp_p,
        lp_kernel: int = lp_kernel,
        lp_stride: int = lp_stride,
        dropout_p: float = dropout_p,
        adaptive_out: int = adaptive_out,
    ):
        super(Model, self).__init__()
        # 2D Lp pooling (p-norm pooling)
        self.lppool = nn.LPPool2d(lp_p, kernel_size=lp_kernel, stride=lp_stride)
        # AlphaDropout maintains self-normalizing activations properties
        self.alpha_dropout = nn.AlphaDropout(dropout_p)
        # 1x1 conv to mix channels and expand representation capacity
        self.conv1x1 = nn.Conv2d(in_channels, in_channels * expand_factor, kernel_size=1, bias=True)
        # Activation
        self.act = nn.GELU()
        # Adaptive 1D max pooling applied on the flattened spatial sequence
        self.adaptive_pool1d = nn.AdaptiveMaxPool1d(adaptive_out)
        # Small epsilon for numerical stability during normalization
        self.eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (N, C_out, adaptive_out) where
                          C_out = in_channels * expand_factor
        """
        # Step 1: Lp pooling to aggregate local neighborhoods (reduces spatial dims)
        x = self.lppool(x)  # (N, C, H1, W1)

        # Step 2: AlphaDropout for regularization (respects self.training)
        x = self.alpha_dropout(x)

        # Step 3: Channel mixing / expansion via 1x1 conv and non-linear activation
        x = self.conv1x1(x)  # (N, C_out, H1, W1)
        x = self.act(x)

        # Step 4: Flatten spatial dims into a sequence dimension for 1D pooling
        N, C_out, H1, W1 = x.shape
        seq_len = H1 * W1
        x = x.view(N, C_out, seq_len)  # (N, C_out, L)

        # Step 5: Adaptive max pool the sequence to fixed-length output
        x = self.adaptive_pool1d(x)  # (N, C_out, adaptive_out)

        # Step 6: L2-normalize across the channel dimension for each output position
        # Norm shape: (N, 1, adaptive_out) to broadcast across channels
        norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x = x / (norm + self.eps)

        return x

def get_inputs():
    """
    Returns a list with one input tensor shaped (batch_size, in_channels, height, width).
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters that could be used to instantiate the model.
    Keeping it empty to mimic common examples; all module-level constants are available.
    """
    return []