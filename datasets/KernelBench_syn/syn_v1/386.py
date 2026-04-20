import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex 3D processing module that:
    - Applies reflection padding to a 5D input (B, C, D, H, W)
    - Performs a local smoothing via average pooling and blends with the padded input
    - Applies a lazily-initialized 3D BatchNorm (nn.LazyBatchNorm3d)
    - Uses Mish activation (nn.Mish)
    - Computes a channel-wise attention vector from global-average-pooled features and a provided scale vector,
      then applies the attention to the activated features and reduces spatially to produce the final output
    """
    def __init__(self, pad: int = 1):
        super(Model, self).__init__()
        # Reflection padding applied equally on all 3 spatial dims
        self.pad = nn.ReflectionPad3d(pad)
        # LazyBatchNorm3d will infer num_features from the first forward pass
        self.bn = nn.LazyBatchNorm3d()
        # Mish activation
        self.act = nn.Mish()
        # Small learnable scalar to modulate attention magnitude
        self.att_scale = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x: torch.Tensor, channel_scale: torch.Tensor) -> torch.Tensor:
        """
        Forward computation:
        1. Reflection pad the input
        2. Smooth via avg_pool3d and blend with padded input
        3. Apply BatchNorm3d (lazy)
        4. Apply Mish activation
        5. Global-average-pool spatial dims to get channel descriptors
        6. Combine descriptors with an external channel_scale vector, produce sigmoid attention
        7. Re-weight feature maps with attention and sum over channel dimension to produce output 4D tensor

        Args:
            x (torch.Tensor): Input of shape (B, C, D, H, W)
            channel_scale (torch.Tensor): 1D tensor of shape (C,) providing a per-channel scaling vector

        Returns:
            torch.Tensor: Output tensor of shape (B, D_p, H_p, W_p) where spatial dims include padding effects
        """
        # 1) Reflection pad
        x_pad = self.pad(x)  # (B, C, D+2*pad, H+2*pad, W+2*pad)

        # 2) Local smoothing with average pooling (keeps same spatial shape due to padding) and blend
        x_smooth = F.avg_pool3d(x_pad, kernel_size=3, stride=1, padding=1)
        x_blend = 0.6 * x_pad + 0.4 * x_smooth  # element-wise blend

        # 3) Batch normalization (lazy initialization happens on first forward)
        x_bn = self.bn(x_blend)

        # 4) Mish activation
        x_act = self.act(x_bn)

        # 5) Global average pooling over spatial dims -> (B, C)
        gap = x_act.mean(dim=[2, 3, 4])

        # 6) Channel attention: combine gap with external channel_scale, apply learnable modulation and sigmoid
        # Ensure channel_scale shape matches channels
        # channel_scale is expected shape (C,)
        att_raw = gap * channel_scale  # (B, C)
        att_raw = att_raw * self.att_scale
        att = torch.sigmoid(att_raw).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1, 1)

        # 7) Re-weight feature maps and reduce over channels to produce final output
        x_weighted = x_act * att  # (B, C, D_p, H_p, W_p)
        out = x_weighted.sum(dim=1)  # (B, D_p, H_p, W_p)

        return out

# Configuration / default sizes
BATCH = 4
CHANNELS = 32
DEPTH = 8
HEIGHT = 16
WIDTH = 16

def get_inputs():
    """
    Creates example inputs:
    - x: random 5D tensor (BATCH, CHANNELS, DEPTH, HEIGHT, WIDTH)
    - channel_scale: random 1D per-channel scaling vector (CHANNELS,)
    """
    x = torch.randn(BATCH, CHANNELS, DEPTH, HEIGHT, WIDTH)
    channel_scale = torch.randn(CHANNELS)
    return [x, channel_scale]

def get_init_inputs():
    """
    No special initialization inputs required (the LazyBatchNorm3d will initialize itself on first forward).
    """
    return []