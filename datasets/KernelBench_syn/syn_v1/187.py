import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    A more complex 1D sequence module that demonstrates a small upsampling encoder
    followed by channel projection and adaptive pooling to a fixed temporal size.

    Architecture:
    - ConvTranspose1d upsampling (deconv1) to expand temporal resolution by `up_scale`
    - GELU activation
    - ConvTranspose1d refinement (deconv2) with kernel_size=3 to mix neighborhoods
    - AdaptiveMaxPool1d to a fixed output length (pool_size)
    - A projected skip connection: pool the original input to pool_size then apply
      a 1x1 ConvTranspose1d (proj) to match output channels
    - Residual add, LayerNorm across channels (applied on permuted tensor),
      and a learned per-channel gating via sigmoid-scaled parameter
    """
    def __init__(self,
                 in_channels: int,
                 mid_channels: int,
                 out_channels: int,
                 up_scale: int,
                 pool_size: int):
        """
        Initializes the layered model.

        Args:
            in_channels: Number of channels in the input tensor.
            mid_channels: Number of channels in the intermediate upsampled representation.
            out_channels: Number of channels for the final output.
            up_scale: Integer upsampling factor performed by the first ConvTranspose1d.
            pool_size: Temporal length after adaptive pooling for both main path and skip.
        """
        super(Model, self).__init__()

        # First transposed convolution upsamples by exactly `up_scale`:
        # out_length = (in_length - 1) * stride + kernel_size -> if kernel_size == stride == up_scale -> out = in * up_scale
        self.deconv1 = nn.ConvTranspose1d(in_channels,
                                          mid_channels,
                                          kernel_size=up_scale,
                                          stride=up_scale,
                                          padding=0,
                                          output_padding=0,
                                          bias=True)

        # Small refinement conv (keeps same temporal length)
        self.deconv2 = nn.ConvTranspose1d(mid_channels,
                                          out_channels,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1,
                                          bias=True)

        # Adaptive pooling to a fixed temporal length
        self.pool = nn.AdaptiveMaxPool1d(pool_size)

        # Project pooled original input channels to match out_channels (1x1 conv transpose acts like conv here)
        self.proj = nn.ConvTranspose1d(in_channels,
                                       out_channels,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       bias=True)

        # Normalization will be applied over the channel dimension after permute
        self.norm = nn.LayerNorm(out_channels)

        # Activation
        self.act = nn.GELU()

        # Learned per-channel gating parameter (broadcast over temporal dimension)
        self.gain = nn.Parameter(torch.randn(out_channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, in_channels, seq_len)

        Returns:
            Tensor of shape (batch, out_channels, pool_size)
        """
        # Main path: upsample -> activate -> refine -> pool
        y = self.deconv1(x)           # (batch, mid_channels, seq_len * up_scale)
        y = self.act(y)
        y = self.deconv2(y)          # (batch, out_channels, seq_len * up_scale)
        y = self.pool(y)             # (batch, out_channels, pool_size)

        # Skip path: pool original input to pool_size then project channels
        skip = self.pool(x)          # (batch, in_channels, pool_size)
        skip = self.proj(skip)       # (batch, out_channels, pool_size)

        # Residual add
        out = y + skip               # (batch, out_channels, pool_size)

        # Apply LayerNorm across channel dim: LayerNorm expects last dim normalized, so permute
        out = out.permute(0, 2, 1)   # (batch, pool_size, out_channels)
        out = self.norm(out)
        out = out.permute(0, 2, 1)   # (batch, out_channels, pool_size)

        # Channel-wise gating with sigmoid against a learned per-channel parameter
        gate = torch.sigmoid(self.gain)  # (out_channels, 1)
        out = out * gate                 # broadcast over temporal axis

        # Final smooth activation
        out = torch.tanh(out)

        return out

# Configuration variables (module-level)
batch_size = 8
in_channels = 16
mid_channels = 32
out_channels = 24
input_length = 128
up_scale = 4        # upsample factor for deconv1 -> output length becomes input_length * up_scale
pool_output_size = 64  # final temporal length after pooling (must be <= input_length)

def get_inputs():
    """
    Returns a list with a single input tensor suitable for the Model above.
    Shape: (batch_size, in_channels, input_length)
    """
    x = torch.randn(batch_size, in_channels, input_length)
    return [x]

def get_init_inputs():
    """
    Return initialization parameters used to construct the Model.
    These values correspond to the signature of Model(...) in this file.
    """
    return [in_channels, mid_channels, out_channels, up_scale, pool_output_size]