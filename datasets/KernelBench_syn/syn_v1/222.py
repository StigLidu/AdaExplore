import torch
import torch.nn as nn
import torch.nn.functional as F

# Module-level configuration
batch_size = 8
channels = 256
height = 64
width = 64
spatial_pool_size = (4, 4)  # Adaptive pooling target for spatial compression

class Model(nn.Module):
    """
    Complex image feature compressor that:
    - Compresses spatial information with AdaptiveAvgPool2d
    - Applies a bottleneck MLP (two Linear layers) on flattened pooled features
    - Computes a channel-wise gating vector from global (1x1) pooled features
    - Applies Hardswish nonlinearities and fuses gated features
    - Upsamples back to original spatial resolution and adds a residual connection
    """
    def __init__(self, in_channels: int = channels, pool_size: tuple = spatial_pool_size):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.pool_size = pool_size

        # Spatial pooling to reduce HxW to a small fixed grid (e.g., 4x4)
        self.pool_spatial = nn.AdaptiveAvgPool2d(self.pool_size)

        # Global channel pooling to (1,1) to compute per-channel gates
        self.pool_channel = nn.AdaptiveAvgPool2d((1, 1))

        # Dimensions for the bottleneck MLP that operates on flattened pooled features
        flat_dim = in_channels * self.pool_size[0] * self.pool_size[1]
        hidden_dim = max(flat_dim // 4, 64)  # compress then expand; ensure reasonable minimum

        # Bottleneck MLP applied to flattened spatial-pooled features
        self.fc1 = nn.Linear(flat_dim, hidden_dim, bias=True)
        self.act = nn.Hardswish()
        self.fc2 = nn.Linear(hidden_dim, flat_dim, bias=True)

        # Small gating MLP per channel (global context to channel-wise scale)
        self.gate_fc = nn.Linear(in_channels, in_channels, bias=True)

        # Initialize weights with a stable scheme
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='linear')
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='linear')
        nn.init.zeros_(self.fc2.bias)
        nn.init.kaiming_normal_(self.gate_fc.weight, nonlinearity='linear')
        nn.init.zeros_(self.gate_fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        1. Spatially pool x to small grid -> x_pool (N, C, pH, pW)
        2. Flatten and run through bottleneck MLP -> x_mlp (N, C * pH * pW)
        3. Reshape back to (N, C, pH, pW)
        4. Compute channel gates from global pooled x -> g (N, C), apply Hardswish
        5. Broadcast gates and multiply with x_mlp
        6. Upsample to original (H, W) and add residual connection

        Args:
            x: Input tensor of shape (N, C, H, W)
        Returns:
            Tensor of shape (N, C, H, W)
        """
        N, C, H, W = x.shape

        # 1) Spatial compression
        x_pool = self.pool_spatial(x)  # (N, C, pH, pW)

        # 2) Bottleneck MLP on flattened pooled features
        x_flat = x_pool.view(N, -1)  # (N, C * pH * pW)
        hidden = self.fc1(x_flat)    # (N, hidden_dim)
        hidden = self.act(hidden)
        x_back = self.fc2(hidden)    # (N, C * pH * pW)
        x_back = x_back.view(N, C, self.pool_size[0], self.pool_size[1])  # (N, C, pH, pW)

        # 3) Channel gating from global context
        g = self.pool_channel(x).view(N, C)  # (N, C)
        g = self.gate_fc(g)                   # (N, C)
        g = self.act(g)                       # Hardswish gating, (N, C)
        g = g.view(N, C, 1, 1)                # (N, C, 1, 1)

        # 4) Apply gating to reconstructed pooled features
        gated = x_back * g  # broadcast multiply -> (N, C, pH, pW)

        # 5) Upsample to original resolution and add residual (with a learned scale of 0.5)
        up = F.interpolate(gated, size=(H, W), mode='bilinear', align_corners=False)  # (N, C, H, W)
        out = up + 0.5 * x  # residual fusion

        return out

def get_inputs():
    """
    Produces a batch of random image tensors with the configured shapes.

    Returns:
        list: [x] where x is of shape (batch_size, channels, height, width)
    """
    x = torch.randn(batch_size, channels, height, width)
    return [x]

def get_init_inputs():
    """
    No special initialization inputs required for this model.

    Returns:
        list: empty
    """
    return []