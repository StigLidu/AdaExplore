import torch
import torch.nn as nn

# Configuration / shape parameters
batch_size = 8
in_channels = 16
height = 32  # must be divisible by 2 for pooling/unpooling
width = 32   # must be divisible by 2 for pooling/unpooling

class Model(nn.Module):
    """
    A moderately complex module that demonstrates a mixed 2D <-> 1D processing pattern:
      - Applies MaxPool2d (with indices) to reduce spatial resolution.
      - Flattens pooled spatial map into a sequence and applies LazyInstanceNorm1d + LazyConv1d
        to process spatial locations as a temporal sequence (per channel).
      - Restores the spatial layout and uses MaxUnpool2d (with the original indices) to
        reconstruct to the original resolution.
      - Uses a learned channel-wise scale and a residual connection to produce the final output.

    This pattern combines pooling/unpooling semantics with lazy initialization layers
    that set their shapes at first forward pass.
    """
    def __init__(self):
        super(Model, self).__init__()
        # 2x downsampling / upsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        # Lazy normalization & convolution in 1D over spatial locations per-channel.
        # We use Lazy modules so in_channels / num_features are inferred at first forward.
        # out_channels is set equal to in_channels to keep channels consistent for unpooling.
        self.norm1d = nn.LazyInstanceNorm1d(affine=True)
        self.conv1d = nn.LazyConv1d(out_channels=in_channels, kernel_size=3, padding=1, bias=True)

        # Learnable per-channel scale applied after unpooling (keeps shape broadcastable)
        # Initialized as ones so initial behavior is near identity.
        self.register_parameter("channel_scale", nn.Parameter(torch.ones(1, in_channels, 1, 1)))

        # Small final non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
          x: (B, C, H, W)
        Returns:
          out: (B, C, H, W) same spatial shape as input
        """
        # Save residual for later
        residual = x

        # 1) Downsample with indices
        pooled, indices = self.pool(x)  # pooled: (B, C, H//2, W//2)

        # 2) Flatten spatial dims to a sequence: (B, C, L) where L = (H//2)*(W//2)
        b, c, h2, w2 = pooled.shape
        seq_len = h2 * w2
        pooled_seq = pooled.view(b, c, seq_len)  # shape suitable for InstanceNorm1d / Conv1d

        # 3) Normalize across channels per sequence
        normed = self.norm1d(pooled_seq)  # LazyInstanceNorm1d infers num_features = c

        # 4) 1D convolution over the spatial sequence (per-channel receptive field along sequence)
        conv_out = self.conv1d(normed)  # LazyConv1d infers in_channels and produces out_channels==in_channels

        # 5) Non-linearity
        conv_out = torch.relu(conv_out)

        # 6) Restore spatial layout to (B, C, H//2, W//2)
        processed_pooled = conv_out.view(b, c, h2, w2)

        # 7) Unpool back to original resolution using original indices
        unpooled = self.unpool(processed_pooled, indices, output_size=residual.shape)  # (B, C, H, W)

        # 8) Apply learned per-channel scaling and add residual (skip connection)
        scaled = unpooled * self.channel_scale  # broadcast over spatial dims
        out = residual + scaled

        # 9) Final activation
        out = self.act(out)

        return out

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    # No special external initialization parameters required (lazy layers initialize at first forward).
    return []