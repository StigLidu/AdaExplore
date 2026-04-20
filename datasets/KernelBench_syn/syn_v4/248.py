import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

"""
Complex PyTorch kernel module combining Conv2d, AdaptiveMaxPool2d, and Softmax.

Design:
- Two-stage convolutional transform (3x3 then 1x1) to build channel features.
- Adaptive max pooling to produce a compact channel descriptor.
- Softmax over channels to create a channel-wise attention vector.
- Channel-wise scaling of convolution output (attention gating).
- Residual projection of the input (1x1 conv) to match output channels and pooled spatial size.
- Final global average over spatial dims to produce a compact output per channel.

Module-level configuration variables are provided for easy adjustments.
"""

# Configuration
BATCH_SIZE = 8
IN_CHANNELS = 3
MID_CHANNELS = 64
OUT_CHANNELS = 128
HEIGHT = 128
WIDTH = 128
POOL_OUTPUT = (1, 1)  # Adaptive pooling target (H_out, W_out)
SOFTMAX_DIM = 1  # Softmax applied over channel dimension after pooling

class Model(nn.Module):
    """
    Multi-stage convolutional module with adaptive pooling and channel attention.

    Architecture:
      x -> Conv2d(in->mid, kernel=3,pad=1) -> ReLU
        -> Conv2d(mid->out, kernel=1) -> AdaptiveMaxPool2d(pool_size)
        -> channel descriptor -> Softmax -> channel attention
        -> scale original conv2 output by attention
        -> add residual projected input (pooled & 1x1 conv)
        -> spatial global average -> output (batch, out_channels)
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        pool_size: Tuple[int, int] = (1, 1),
        softmax_dim: int = 1,
    ):
        """
        Initializes the model.

        Args:
            in_channels (int): Number of input channels.
            mid_channels (int): Number of intermediate convolution channels.
            out_channels (int): Number of output channels produced by the conv stack.
            pool_size (tuple): Target spatial size (H_out, W_out) for AdaptiveMaxPool2d.
            softmax_dim (int): Dimension over which to apply Softmax (typically channel dim).
        """
        super(Model, self).__init__()
        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=True)

        # Residual projection to match out_channels and pooled spatial dims
        self.res_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

        # Adaptive pooling and softmax for channel attention
        self.pool = nn.AdaptiveMaxPool2d(pool_size)
        self.softmax = nn.Softmax(dim=softmax_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_channels, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch, out_channels) after global averaging.
        """
        # Stage 1: convolutional feature extraction
        feat = self.conv1(x)               # (B, mid, H, W)
        feat = F.relu(feat)                # non-linearity

        # Stage 2: channel expansion
        feat_out = self.conv2(feat)        # (B, out, H, W)

        # Descriptor: adaptive max pooling to compact spatial info
        desc = self.pool(feat_out)         # (B, out, pool_h, pool_w)

        # Collapse spatial dims of descriptor into a single channel descriptor vector
        # If pool_size > (1,1), average across those spatial positions
        desc_flat = desc.view(desc.size(0), desc.size(1), -1)  # (B, out, pool_h * pool_w)
        desc_vec = desc_flat.mean(dim=2)                       # (B, out)

        # Channel attention via softmax across channels
        attn = self.softmax(desc_vec)                          # (B, out)
        attn = attn.view(desc.size(0), desc.size(1), 1, 1)     # (B, out, 1, 1)

        # Apply channel-wise attention to convolutional output
        gated = feat_out * attn                                # (B, out, H, W)

        # Residual branch: project original input to match out channels & pool spatially
        res_pooled = F.adaptive_max_pool2d(x, self.pool.output_size if hasattr(self.pool, "output_size") else POOL_OUTPUT)
        # Fallback to configured POOL_OUTPUT if attribute not available; ensure correct spatial size
        if res_pooled.shape[2:] != desc.shape[2:]:
            # enforce the expected pool shape
            res_pooled = F.adaptive_max_pool2d(x, desc.shape[2:])
        res_proj = self.res_proj(res_pooled)                   # (B, out, pool_h, pool_w)

        # If residual projection has smaller spatial size than gated, upsample to match gated
        if res_proj.shape[2:] != gated.shape[2:]:
            res_proj_upsampled = F.interpolate(res_proj, size=gated.shape[2:], mode='nearest')
        else:
            res_proj_upsampled = res_proj

        # Combine gated features with residual projection
        combined = gated + res_proj_upsampled                  # (B, out, H, W)

        # Final aggregation: global average over spatial dims -> compact vector per channel
        out = combined.mean(dim=(2, 3))                        # (B, out)
        return out

def get_inputs() -> List[torch.Tensor]:
    """
    Returns example input tensors for the model run.

    Returns:
        List[torch.Tensor]: [x] where x has shape (BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)
    """
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)
    return [x]

def get_init_inputs() -> List:
    """
    Returns initialization parameters matching Model.__init__ signature.

    Returns:
        List: [in_channels, mid_channels, out_channels, pool_size, softmax_dim]
    """
    return [IN_CHANNELS, MID_CHANNELS, OUT_CHANNELS, POOL_OUTPUT, SOFTMAX_DIM]