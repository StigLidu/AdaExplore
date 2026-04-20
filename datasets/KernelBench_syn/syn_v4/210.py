import torch
import torch.nn as nn
import torch.nn.functional as F

# Configuration
batch_size = 16
in_channels = 64
mid_channels = 128
out_channels = 32
H = 32
W = 32

class Model(nn.Module):
    """
    A moderately complex upsampling module that:
      - Uses ConvTranspose2d to perform learned upsampling in two stages
      - Applies LazyBatchNorm2d after each transposed convolution (lazy init)
      - Uses LeakyReLU non-linearity
      - Builds an attention-like channel gating from global context
      - Adds a learned skip connection (conv transpose) from the input to the upsampled output

    Forward graph (high level):
      x -> convt1 -> bn1 -> act -> convt2 -> bn2 -> act -> (global context gating)
      input -> conv_skip -> (upsampled skip)
      out = gated_features + upsampled_skip
    """
    def __init__(self,
                 in_ch: int = in_channels,
                 mid_ch: int = mid_channels,
                 out_ch: int = out_channels,
                 negative_slope: float = 0.2):
        super(Model, self).__init__()
        # First learned upsampling: doubles spatial resolution
        self.convt1 = nn.ConvTranspose2d(in_channels=in_ch,
                                         out_channels=mid_ch,
                                         kernel_size=4,
                                         stride=2,
                                         padding=1,
                                         bias=False)
        # Second conv keeping the upsampled resolution but projecting to out_ch
        self.convt2 = nn.ConvTranspose2d(in_channels=mid_ch,
                                         out_channels=out_ch,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1,
                                         bias=False)
        # Lazy batch norms - they'll infer num_features at first forward pass
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()
        # Learned skip connection: project & upsample input to match output resolution & channels
        self.conv_skip = nn.ConvTranspose2d(in_channels=in_ch,
                                            out_channels=out_ch,
                                            kernel_size=2,
                                            stride=2,
                                            padding=0,
                                            bias=False)
        # Activation
        self.act = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, in_channels, H, W)

        Returns:
            Tensor of shape (B, out_channels, H*2, W*2)
        """
        # Learned upsample stage 1
        y = self.convt1(x)            # (B, mid_ch, H*2, W*2)
        y = self.bn1(y)
        y = self.act(y)

        # Learned upsample stage 2 / projection
        y = self.convt2(y)            # (B, out_ch, H*2, W*2)
        y = self.bn2(y)
        y = self.act(y)

        # Global context: channel-wise summary -> gating
        # shape -> (B, out_ch, 1, 1)
        context = y.mean(dim=(2, 3), keepdim=True)
        gating = torch.sigmoid(context)

        # Apply gating (channel-wise attention)
        y_gated = y * gating

        # Learned skip connection: project and upsample original input
        skip = self.conv_skip(x)      # (B, out_ch, H*2, W*2)

        # Combine gated path with skip (residual style)
        out = y_gated + skip

        # Final non-linearity (light touch)
        out = self.act(out)

        return out

def get_inputs():
    """
    Generates a single input tensor for the model.

    Returns:
        list: [x] where x has shape (batch_size, in_channels, H, W)
    """
    x = torch.randn(batch_size, in_channels, H, W, dtype=torch.float32)
    return [x]

def get_init_inputs():
    """
    Returns initialization inputs if needed. This model does not require
    extra runtime initialization parameters (LazyBatchNorm2d will initialize
    at first forward).

    Returns:
        list: Empty list.
    """
    return []