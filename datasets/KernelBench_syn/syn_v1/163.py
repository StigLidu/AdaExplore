import torch
import torch.nn as nn
from typing import Tuple, List

class Model(nn.Module):
    """
    Complex model that combines 3D max-pooling, reshaping to merge depth into channels,
    2D constant padding, 2D channel-wise dropout, channel scaling, and spatial global averaging.

    Pipeline:
      1. MaxPool3d over (D, H, W)
      2. Merge depth dimension into the channel dimension -> (N, C * D', H', W')
      3. ConstantPad2d around spatial dims (H', W')
      4. Dropout2d to randomly zero entire channels
      5. Multiply by a provided channel-wise scale vector (broadcasted)
      6. Global spatial average over H and W -> (N, C * D')
    """
    def __init__(self, pool_kernel: Tuple[int, int, int], dropout_prob: float, pad: int):
        """
        Args:
            pool_kernel: 3-tuple specifying kernel size (kD, kH, kW) for MaxPool3d.
            dropout_prob: Probability for nn.Dropout2d.
            pad: Integer padding to apply on all four sides (left, right, top, bottom).
        """
        super(Model, self).__init__()
        self.pool_kernel = pool_kernel
        self.pool = nn.MaxPool3d(kernel_size=pool_kernel, stride=pool_kernel)
        # ConstantPad2d expects (left, right, top, bottom)
        pads = (pad, pad, pad, pad)
        self.pad2d = nn.ConstantPad2d(pads, 0.0)  # pad value will be applied via multiplication later if needed
        self.dropout2d = nn.Dropout2d(p=dropout_prob)

    def forward(self, x: torch.Tensor, channel_scale: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (N, C, D, H, W)
            channel_scale: 1D tensor of length (C * D') where D' = D // pool_kernel[0]

        Returns:
            Tensor of shape (N, C * D') with spatially averaged, scaled channel responses.
        """
        # 1) 3D max pooling
        x_p = self.pool(x)  # (N, C, D', H', W')

        N, C, Dp, Hp, Wp = x_p.shape

        # 2) Merge depth into channels -> (N, C * D', H', W')
        # Ensure contiguous memory for view
        x_merged = x_p.contiguous().view(N, C * Dp, Hp, Wp)

        # 3) Constant padding on spatial dims
        x_padded = self.pad2d(x_merged)  # (N, C * D', H'+2*pad, W'+2*pad)

        # 4) Dropout2d over channels
        x_dropped = self.dropout2d(x_padded)

        # 5) Channel-wise scaling: channel_scale -> (1, C*D', 1, 1) to broadcast
        expected_ch = C * Dp
        if channel_scale.numel() != expected_ch:
            raise ValueError(f"channel_scale length ({channel_scale.numel()}) does not match expected channels ({expected_ch})")
        scale = channel_scale.view(1, expected_ch, 1, 1)
        x_scaled = x_dropped * scale  # broadcast multiplication

        # 6) Global spatial average over H and W -> (N, C * D')
        out = x_scaled.mean(dim=(2, 3))

        return out


# Configuration / default inputs for this kernel module
BATCH_SIZE = 8
CHANNELS = 16
DEPTH = 8        # must be divisible by POOL_KERNEL[0]
HEIGHT = 32
WIDTH = 32

POOL_KERNEL = (2, 2, 2)   # (kD, kH, kW)
DROPOUT_PROB = 0.25
PAD_VAL = 1               # integer padding amount (applied symmetrically)

def get_inputs() -> List[torch.Tensor]:
    """
    Create input tensors consistent with the configuration above.

    Returns:
        [x, channel_scale]
          - x: Tensor of shape (BATCH_SIZE, CHANNELS, DEPTH, HEIGHT, WIDTH)
          - channel_scale: 1D tensor of length CHANNELS * (DEPTH // POOL_KERNEL[0])
    """
    x = torch.randn(BATCH_SIZE, CHANNELS, DEPTH, HEIGHT, WIDTH)
    Dp = DEPTH // POOL_KERNEL[0]
    channel_scale = torch.randn(CHANNELS * Dp)
    return [x, channel_scale]

def get_init_inputs() -> List:
    """
    Returns initialization parameters for the Model constructor in the order:
      [pool_kernel, dropout_prob, pad]
    """
    return [POOL_KERNEL, DROPOUT_PROB, PAD_VAL]