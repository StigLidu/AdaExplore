import torch
import torch.nn as nn

"""
Complex composite module combining GroupNorm, RMSNorm, and Mish activation with
a channel gating mechanism and residual connection. The model accepts a 4D image-like
tensor and a channel gating tensor, applies group normalization, uses a learned
1x1 convolution to mix channels, gates channels via an RMS-normalized vector, and
finally projects the resulting feature map to a compact per-sample representation.
"""

# Configuration
batch_size = 8
channels = 64
height = 16
width = 16
num_groups = 8  # must divide channels

class Model(nn.Module):
    """
    Composite model that demonstrates a multi-stage processing pipeline:
      - Group Normalization over spatial inputs
      - Channel-wise gating computed from an external vector normalized by RMSNorm
      - 1x1 convolution for channel mixing
      - Mish activation and residual connection
      - Final projection to a compact embedding per sample

    Forward signature: forward(x, gate)
      x: Tensor of shape (N, C, H, W)
      gate: Tensor of shape (N, C)
    """
    def __init__(self, num_groups: int = num_groups, channels: int = channels,
                 height: int = height, width: int = width):
        super(Model, self).__init__()
        if channels % num_groups != 0:
            raise ValueError(f"num_groups ({num_groups}) must divide channels ({channels})")

        # Normalization layers
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        # RMSNorm applied to the gating vector (normalized_shape == number of channels)
        self.rms_norm = nn.RMSNorm(normalized_shape=channels, eps=1e-6)

        # Non-linearity
        self.mish = nn.Mish()

        # Channel mixing via 1x1 convolution (keeps spatial dims)
        self.conv1x1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, bias=True)

        # Final linear projection from flattened feature map to channel-sized embedding
        self.height = height
        self.width = width
        flat_dim = channels * height * width
        self.proj = nn.Linear(flat_dim, channels, bias=True)

        # Small epsilon to stabilize norms where needed
        self._eps = 1e-6

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the composite module.

        Steps:
          1. Apply GroupNorm to spatial tensor x -> x_norm
          2. Compute an average-pooled descriptor of x_norm across spatial dims
          3. Normalize the external gate vector via RMSNorm and squash with sigmoid to get per-channel gates
          4. Scale x_norm by the per-channel gates
          5. Mix channels with a 1x1 convolution, apply Mish activation
          6. Add residual connection from the original x, then flatten and project to per-sample embedding

        Returns:
          Tensor of shape (N, channels)
        """
        # x: (N, C, H, W)
        # gate: (N, C)
        if x.dim() != 4:
            raise ValueError("x must be a 4D tensor (N, C, H, W)")
        if gate.dim() != 2:
            raise ValueError("gate must be a 2D tensor (N, C)")

        # 1) Group Normalization over channels for spatial input
        x_norm = self.group_norm(x)

        # 2) Spatial summary (not used to gate directly, but could be helpful for other computations)
        # keep as an informative operation to increase computational pattern complexity
        spatial_desc = torch.mean(x_norm, dim=(-2, -1), keepdim=True)  # (N, C, 1, 1)

        # 3) Normalize the gate vector and produce gating weights in (0,1)
        gate_normed = self.rms_norm(gate)  # (N, C)
        gate_weights = torch.sigmoid(gate_normed).unsqueeze(-1).unsqueeze(-1)  # (N, C, 1, 1)

        # 4) Channel-wise gating of the normalized spatial features
        x_gated = x_norm * gate_weights  # (N, C, H, W)

        # 5) 1x1 convolution to mix channels, then Mish activation
        mixed = self.conv1x1(x_gated)  # (N, C, H, W)
        activated = self.mish(mixed)

        # 6) Residual connection (adds original input's information)
        # Match shapes and add; here we use the original input x to form a residual.
        # To avoid scale mismatch, normalize residual by its Frobenius norm per sample.
        # Compute per-sample norm over (C,H,W)
        residual_norm = torch.norm(x.view(x.size(0), -1), p=2, dim=1, keepdim=True)  # (N, 1)
        residual_norm = residual_norm.clamp(min=self._eps).unsqueeze(-1)  # (N,1,1) for broadcasting safety
        # reshape for broadcasting across channels and spatial dims
        residual_scale = (1.0 / residual_norm).unsqueeze(-1)  # (N,1,1,1)
        x_rescaled = x * residual_scale  # scaled residual to keep numerical stability

        out = x_rescaled + activated  # (N, C, H, W)

        # 7) Flatten spatial dims and project to a compact embedding per sample
        out_flat = out.view(out.size(0), -1)  # (N, C*H*W)
        embedding = self.proj(out_flat)  # (N, C)

        return embedding

# Input generation utilities
def get_inputs():
    """
    Returns:
        list: [x, gate] where
          x is a random tensor with shape (batch_size, channels, height, width)
          gate is a random tensor with shape (batch_size, channels)
    """
    x = torch.randn(batch_size, channels, height, width)
    gate = torch.randn(batch_size, channels)
    return [x, gate]

def get_init_inputs():
    """
    Returns initialization parameters that can be used to construct the Model:
      [num_groups, channels, height, width]
    """
    return [num_groups, channels, height, width]