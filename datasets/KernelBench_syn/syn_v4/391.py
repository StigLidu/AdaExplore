import torch
import torch.nn as nn
import torch.nn.functional as F

# Configuration
batch_size = 8
in_channels = 64
height = 128
width = 128
pool_output_size = (8, 8)  # (pH, pW)
reduction_ratio = 4        # for channel bottleneck in attention

class Model(nn.Module):
    """
    Complex image-processing module that combines adaptive spatial pooling,
    instance normalization over flattened spatial dimensions, ReLU6 activations,
    and a small channel-attention MLP. The module returns an output with the
    same spatial dimensions and channels as the input, with a residual connection.
    """
    def __init__(self, channels: int = in_channels, pool_size: tuple = pool_output_size, reduction: int = reduction_ratio):
        """
        Initializes layers:
            - AdaptiveAvgPool2d to reduce spatial dims to pool_size
            - InstanceNorm1d applied on (N, C, L) where L = pooled spatial size
            - ReLU6 activation
            - Two Linear layers to form a small channel-attention bottleneck MLP
        Args:
            channels: number of input channels
            pool_size: target (height, width) for adaptive pooling
            reduction: channel reduction ratio for the attention MLP
        """
        super(Model, self).__init__()
        self.channels = channels
        self.pool_size = pool_size
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        # InstanceNorm1d normalizes per-instance across the spatial (L) dimension when input shaped (N, C, L)
        self.inst_norm = nn.InstanceNorm1d(num_features=channels, affine=True)
        self.relu6 = nn.ReLU6()
        hidden = max(4, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
            1. Adaptive average pool to (pH, pW)
            2. Flatten spatial dims -> (N, C, L) and apply InstanceNorm1d
            3. ReLU6 activation
            4. Spatial-average to get channel descriptors (N, C)
            5. Small MLP (fc1 -> ReLU6 -> fc2 -> sigmoid) to produce channel gates
            6. Re-scale normalized spatial features with gates, reshape back to (N, C, pH, pW)
            7. Upsample to original spatial size and add residual connection
        """
        N, C, H, W = x.shape
        # 1) Adaptive pooling
        pooled = self.pool(x)  # (N, C, pH, pW)
        pH, pW = self.pool_size

        # 2) Flatten spatial dims and apply InstanceNorm1d
        flat = pooled.view(N, C, pH * pW)  # (N, C, L)
        normed = self.inst_norm(flat)      # (N, C, L)

        # 3) Non-linearity
        activated = self.relu6(normed)     # (N, C, L)

        # 4) Channel descriptor via spatial mean
        chan_desc = activated.mean(dim=2)  # (N, C)

        # 5) Channel-attention MLP
        attn = self.fc1(chan_desc)         # (N, hidden)
        attn = self.relu6(attn)
        attn = self.fc2(attn)              # (N, C)
        attn = torch.sigmoid(attn).unsqueeze(2)  # (N, C, 1)

        # 6) Scale spatial features and reshape back
        scaled = activated * attn          # (N, C, L)
        scaled_spatial = scaled.view(N, C, pH, pW)  # (N, C, pH, pW)

        # 7) Upsample to original spatial resolution and add residual
        up = F.interpolate(scaled_spatial, size=(H, W), mode='bilinear', align_corners=False)
        out = up + x  # residual connection

        return out

def get_inputs():
    """
    Returns a list with a single input tensor shaped (batch_size, in_channels, height, width).
    """
    x = torch.randn(batch_size, in_channels, height, width, dtype=torch.float32)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for the Model constructor.
    """
    return [in_channels, pool_output_size, reduction_ratio]