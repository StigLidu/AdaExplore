import torch
import torch.nn as nn
from typing import List, Optional

"""
Complex model combining LazyInstanceNorm3d, RReLU and InstanceNorm1d with a small
channel-wise gating mechanism. The model demonstrates lazy initialization of
3D instance normalization, a randomized leaky ReLU activation, reshaping to a
sequence for 1D instance normalization, and a lightweight learned gate
constructed on the first forward pass when the channel dimension is known.
"""

# Configuration / default input shape
batch_size = 4
channels = 16    # This can be changed; LazyInstanceNorm3d allows unknown num_features at init
depth = 4
height = 8
width = 8

class Model(nn.Module):
    """
    A model that takes a 5D tensor (N, C, D, H, W), applies 3D instance normalization
    (lazy-initialized), a randomized leaky ReLU, reshapes the spatial volume into a
    per-channel sequence and applies InstanceNorm1d, then computes a small
    channel-wise gating network (created lazily the first time forward is called)
    to modulate the normalized sequence. Residual connection to the original input
    is added at the end.
    """
    def __init__(self):
        super(Model, self).__init__()
        # Lazy 3D instance normalization: num_features will be inferred on first call
        self.inst3d = nn.LazyInstanceNorm3d(affine=True, track_running_stats=False)
        # Randomized leaky ReLU
        self.rrelu = nn.RReLU(lower=0.125, upper=0.333, inplace=False)
        # The following will be created on first forward pass when channel dimension is known
        self.inst1d: Optional[nn.InstanceNorm1d] = None
        self.gate_net: Optional[nn.Module] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation:
          1. Lazy InstanceNorm3d across (C, D, H, W) per-instance.
          2. Randomized leaky ReLU activation.
          3. Flatten spatial dims to sequence of length S = D*H*W -> shape (N, C, S).
          4. InstanceNorm1d across the sequence for each channel (created lazily).
          5. Compute channel-wise gate from sequence summary (mean over S) using a small
             MLP (created lazily) -> produces (N, C) gating values in (0,1).
          6. Apply gate to normalized sequence, reshape back to (N, C, D, H, W).
          7. Add a residual connection to the original input and return.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, D, H, W)

        Returns:
            torch.Tensor: Output tensor of same shape as input.
        """
        # 1. 3D instance normalization (lazy init will infer num_features)
        y = self.inst3d(x)

        # 2. Randomized leaky ReLU
        y = self.rrelu(y)

        N, C, D, H, W = y.shape

        # 3. Flatten spatial dims into a sequence: shape (N, C, S)
        S = D * H * W
        seq = y.view(N, C, S)

        # 4. Lazily create InstanceNorm1d if needed (registers as submodule)
        if self.inst1d is None:
            # create and move to the same device/dtype as input
            self.inst1d = nn.InstanceNorm1d(num_features=C, affine=True, track_running_stats=False)
            self.inst1d.to(device=x.device, dtype=x.dtype)

        seq_norm = self.inst1d(seq)  # shape (N, C, S)

        # 5. Channel-wise gating: summarize sequence -> (N, C), pass through small MLP
        summary = seq_norm.mean(dim=2)  # (N, C)

        if self.gate_net is None:
            # Create a small gating MLP: C -> C//2 -> C (uses ReLU inside)
            hidden = max(4, C // 2)
            # Build as Module so parameters are registered and moved together if needed
            self.gate_net = nn.Sequential(
                nn.Linear(C, hidden, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, C, bias=True)
            )
            self.gate_net.to(device=x.device, dtype=x.dtype)

        gate_logits = self.gate_net(summary)      # (N, C)
        gate = torch.sigmoid(gate_logits).unsqueeze(2)  # (N, C, 1) to broadcast over S

        # 6. Apply gate to normalized sequence and reshape back
        gated_seq = seq_norm * gate             # (N, C, S)
        out = gated_seq.view(N, C, D, H, W)     # (N, C, D, H, W)

        # 7. Residual connection (add original input)
        return out + x

def get_inputs() -> List[torch.Tensor]:
    """
    Returns a list of example input tensors for the module.
    The primary input is a random 5D tensor (batch_size, channels, depth, height, width).
    """
    x = torch.randn(batch_size, channels, depth, height, width)
    return [x]

def get_init_inputs() -> List:
    """
    Returns initialization inputs if needed. This model uses lazy initialization
    and does not require extra initialization parameters.
    """
    return []