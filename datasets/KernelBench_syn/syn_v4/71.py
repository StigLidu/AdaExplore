import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

"""
Complex 3D feature gating module that:
- Applies adaptive 3D average pooling to compress spatial dimensions.
- Uses GELU non-linearity on the pooled features.
- Computes channel-wise descriptors via spatial averaging.
- Passes descriptors through a small MLP (channel mixing).
- Applies FeatureAlphaDropout on the descriptors for robust channel gating.
- Uses sigmoid gates to reweight the pooled feature maps and collapses channels
  to produce a single-channel volumetric output which is then upsampled
  back to the original input resolution.

Structure follows the examples:
- Model class inheriting from nn.Module with __init__ and forward.
- get_inputs() returns example input tensors.
- get_init_inputs() returns initialization parameters used by the Model.
- Configuration variables at module level.
"""

# Configuration variables (can be tuned by tests)
BATCH = 4
CHANNELS = 64
DEPTH = 32
HEIGHT = 32
WIDTH = 16
POOL_OUT = (8, 8, 4)   # (d_out, h_out, w_out)
DROPOUT_P = 0.1        # dropout probability used by FeatureAlphaDropout

class Model(nn.Module):
    """
    3D feature gating model.

    Inputs:
        x: Tensor of shape (B, C, D, H, W)

    Forward pipeline:
        1. AdaptiveAvgPool3d -> (B, C, d_out, h_out, w_out)
        2. GELU activation
        3. Spatial mean -> (B, C)
        4. Two-layer MLP (channel mixing) with GELU in between -> (B, C)
        5. FeatureAlphaDropout applied to descriptors
        6. Sigmoid gating -> broadcast and reweight pooled features
        7. Sum across channels -> (B, d_out, h_out, w_out)
        8. Unsqueeze channel dim and trilinear interpolate back to (D, H, W)
        9. Return (B, 1, D, H, W)
    """
    def __init__(self, pool_output_size: Tuple[int, int, int], dropout_p: float, channels: int):
        super(Model, self).__init__()
        self.pool = nn.AdaptiveAvgPool3d(pool_output_size)
        self.act = nn.GELU()
        # MLP channel reduction and expansion
        hidden = max(1, channels // 2)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=True),
            nn.GELU(),
            nn.Linear(hidden, channels, bias=True)
        )
        # FeatureAlphaDropout for channel descriptors
        self.dropout = nn.FeatureAlphaDropout(p=dropout_p)
        # Keep original channels and pool target for use in forward
        self.channels = channels
        self.pool_output_size = pool_output_size
        self.dropout_p = dropout_p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, D, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (B, 1, D, H, W)
        """
        # Step 1: Adaptive pooling to compress spatial dims
        pooled = self.pool(x)  # (B, C, d_out, h_out, w_out)

        # Step 2: Non-linearity
        pooled_act = self.act(pooled)

        # Step 3: Channel descriptors via spatial average
        # result shape: (B, C)
        descriptors = pooled_act.mean(dim=(2, 3, 4))

        # Step 4: Channel mixing MLP (learned gating scores)
        mixed = self.mlp(descriptors)  # (B, C)

        # Step 5: Apply FeatureAlphaDropout to descriptors for robustness
        dropped = self.dropout(mixed)

        # Step 6: Convert to gates in (0,1)
        gates = torch.sigmoid(dropped)  # (B, C)

        # Step 7: Reweight pooled feature maps by gates (broadcast)
        gates_reshaped = gates.view(gates.size(0), gates.size(1), 1, 1, 1)  # (B, C, 1, 1, 1)
        reweighted = pooled_act * gates_reshaped  # (B, C, d_out, h_out, w_out)

        # Step 8: Collapse channels into a single volumetric feature map
        collapsed = reweighted.sum(dim=1)  # (B, d_out, h_out, w_out)

        # Step 9: Restore a channel dimension and upsample back to original resolution
        collapsed = collapsed.unsqueeze(1)  # (B, 1, d_out, h_out, w_out)
        # Use trilinear interpolation to match original (D, H, W)
        upsampled = F.interpolate(collapsed, size=(x.size(2), x.size(3), x.size(4)),
                                  mode='trilinear', align_corners=False)
        return upsampled  # (B, 1, D, H, W)


def get_inputs() -> List[torch.Tensor]:
    """
    Generates a sample 5D volumetric tensor for testing the module.

    Returns:
        list: [x] where x has shape (BATCH, CHANNELS, DEPTH, HEIGHT, WIDTH)
    """
    x = torch.randn(BATCH, CHANNELS, DEPTH, HEIGHT, WIDTH)
    return [x]

def get_init_inputs() -> List:
    """
    Returns initialization parameters required by the Model constructor.

    Returns:
        list: [pool_output_size (tuple), dropout_p (float), channels (int)]
    """
    return [POOL_OUT, DROPOUT_P, CHANNELS]