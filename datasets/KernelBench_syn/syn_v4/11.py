import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Any, Tuple

class Model(nn.Module):
    """
    Complex 3D spatial refinement module that demonstrates a combination of
    replication padding, local smoothing (average pooling), residual extraction,
    constant padding and a global-local gating mechanism using Tanh.

    The forward pass performs:
      1. Replication pad the input to give boundary context.
      2. Compute a local average (smoothing) and form a residual (detail) by subtracting.
      3. Apply constant padding to the residual to expand receptive field for gating.
      4. Compute a global channel-wise statistic and add it back (broadcast) to create
         a combined feature map.
      5. Apply Tanh nonlinearity, then compute a channel gating factor and modulate
         the features.
      6. Return the refined tensor (same dims as the padded result) ready for further processing.
    """
    def __init__(self, rep_pad: Tuple[int, int, int, int, int, int],
                 const_pad: Tuple[int, int, int, int, int, int],
                 const_val: float,
                 pool_kernel: int):
        """
        Args:
            rep_pad (tuple): 6-int padding for ReplicationPad3d:
                             (w_left, w_right, h_top, h_bottom, d_front, d_back)
            const_pad (tuple): 6-int padding for ConstantPad3d (same ordering).
            const_val (float): Constant value used by ConstantPad3d.
            pool_kernel (int): Kernel size for local average pooling (must be odd for symmetric behavior).
        """
        super(Model, self).__init__()
        self.rep_pad = nn.ReplicationPad3d(rep_pad)
        self.const_pad = nn.ConstantPad3d(const_pad, const_val)
        self.tanh = nn.Tanh()
        self.pool_kernel = pool_kernel
        # Precompute padding for avg_pool3d to preserve spatial dims when stride=1
        self.pool_padding = pool_kernel // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, D, H, W).

        Returns:
            torch.Tensor: Refined tensor after padding, residual extraction, gating.
                          Shape depends on provided paddings: it will be the shape of
                          const_pad(rep_pad(x)).
        """
        # Step 1: replication padding to provide boundary context
        x_rep = self.rep_pad(x)  # (B, C, D1, H1, W1)

        # Step 2: local smoothing (average pooling) to get low-frequency component
        # Use stride=1 to keep same spatial dimensions as x_rep
        local_avg = F.avg_pool3d(x_rep, kernel_size=self.pool_kernel, stride=1, padding=self.pool_padding)

        # Step 3: residual (detail) extraction
        residual = x_rep - local_avg  # high-frequency details

        # Step 4: expand boundary with a constant pad (e.g., to allow larger receptive gating)
        expanded = self.const_pad(residual)  # (B, C, D2, H2, W2)

        # Step 5: compute a global channel-wise statistic (squeeze) and broadcast back
        # This gives each channel a global bias for the gating computation.
        channel_global = torch.mean(expanded, dim=(2, 3, 4), keepdim=True)  # (B, C, 1, 1, 1)

        # Combine local (expanded) and global (channel_global) features
        combined = expanded + channel_global  # broadcast addition

        # Step 6: non-linear gating with Tanh followed by a channel gating factor
        gated = self.tanh(combined)

        # Channel gating: compute a sigmoid of the channel-pooled activations to form multiplicative gates
        channel_gate = torch.sigmoid(torch.mean(gated, dim=(2, 3, 4), keepdim=True))  # (B, C, 1, 1, 1)

        # Modulate the gated features with the channel gates
        refined = gated * channel_gate

        # Return the refined features; shape is (B, C, D_out, H_out, W_out)
        return refined

# Configuration / default parameters
batch_size = 4
channels = 8
depth = 12
height = 16
width = 16

# Padding definitions follow (left, right, top, bottom, front, back)
replication_padding = (1, 1, 2, 2, 0, 1)   # used for ReplicationPad3d
constant_padding = (0, 0, 1, 1, 1, 1)      # used for ConstantPad3d
constant_value = 0.25                       # value to pad with for ConstantPad3d
pool_kernel_size = 3                         # kernel size for avg pooling

def get_inputs() -> List[torch.Tensor]:
    """
    Generates a random 5D input tensor suitable for the Model.

    Returns:
        list: single-entry list with input tensor of shape (batch_size, channels, depth, height, width)
    """
    x = torch.randn(batch_size, channels, depth, height, width)
    return [x]

def get_init_inputs() -> List[Any]:
    """
    Returns the initialization parameters for the Model in the same order
    as the Model.__init__ signature.

    Returns:
        list: [replication_padding, constant_padding, constant_value, pool_kernel_size]
    """
    return [replication_padding, constant_padding, constant_value, pool_kernel_size]