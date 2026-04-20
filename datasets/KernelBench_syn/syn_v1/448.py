import torch
import torch.nn as nn
from typing import Tuple, Optional, List

class Model(nn.Module):
    """
    Complex model that mixes 3D max pooling, spatial padding, 1D channel dropout (on flattened spatial dims),
    and a learnable 1x1 convolution to fuse channels. The computation flow:

    1. MaxPool3d over (D, H, W)
    2. Reduce depth dimension by taking the maximum across the pooled depth (torch.max)
    3. Apply ConstantPad2d to pad spatial dimensions (H, W)
    4. Flatten spatial dims into a sequence and apply Dropout1d across channels/sequence
    5. Restore spatial shape and apply a 1x1 Conv2d to mix channel information
    6. Global average over spatial dims to produce final per-channel features
    """
    def __init__(
        self,
        channels: int,
        pool_kernel: Tuple[int, int, int],
        pool_stride: Optional[Tuple[int, int, int]] = None,
        pad: Tuple[int, int, int, int] = (0, 0, 0, 0),
        pad_value: float = 0.0,
        dropout_p: float = 0.5
    ):
        """
        Args:
            channels: Number of input channels.
            pool_kernel: 3-tuple kernel size for MaxPool3d (D, H, W).
            pool_stride: Optional 3-tuple stride for MaxPool3d. If None, uses pool_kernel.
            pad: 4-tuple (left, right, top, bottom) for ConstantPad2d.
            pad_value: Value used for padding.
            dropout_p: Probability for Dropout1d.
        """
        super(Model, self).__init__()
        if pool_stride is None:
            pool_stride = pool_kernel

        self.pool3d = nn.MaxPool3d(kernel_size=pool_kernel, stride=pool_stride)
        self.pad2d = nn.ConstantPad2d(pad, pad_value)
        # Dropout1d expects input of shape (N, C, L); we'll flatten (H*W) into L
        self.dropout1d = nn.Dropout1d(p=dropout_p)
        # 1x1 convolution to mix channels after spatial ops
        self.channel_fusion = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, bias=True)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (N, C, D, H, W)

        Returns:
            Tensor of shape (N, C) containing per-channel pooled features.
        """
        # 1) 3D max pooling -> (N, C, D', H', W')
        pooled = self.pool3d(x)

        # 2) Reduce along depth dimension by taking maximum across depth -> (N, C, H', W')
        reduced, _ = torch.max(pooled, dim=2)

        # 3) Constant padding on spatial dims -> (N, C, Hp, Wp)
        padded = self.pad2d(reduced)

        # 4) Flatten spatial dims into length L = Hp * Wp -> shape (N, C, L)
        N, C, Hp, Wp = padded.shape
        seq = padded.view(N, C, Hp * Wp)

        # 5) Apply Dropout1d across channels/sequence
        dropped = self.dropout1d(seq)

        # 6) Restore spatial dims -> (N, C, Hp, Wp)
        restored = dropped.view(N, C, Hp, Wp)

        # 7) 1x1 conv to mix channels and a non-linearity -> (N, C, Hp, Wp)
        fused = self.activation(self.channel_fusion(restored))

        # 8) Global average pooling across spatial dims -> (N, C)
        out = fused.mean(dim=(2, 3))

        return out

# Module-level configuration (shapes and initialization parameters)
BATCH_SIZE = 8
CHANNELS = 16
DEPTH = 8
HEIGHT = 32
WIDTH = 32

# Initialization parameters for the model
POOL_KERNEL = (2, 3, 3)          # (D, H, W)
POOL_STRIDE = (2, 3, 3)          # same as kernel here
PAD = (1, 1, 2, 2)               # (left, right, top, bottom)
PAD_VALUE = 0.1
DROPOUT_P = 0.25

def get_inputs() -> List[torch.Tensor]:
    """
    Returns the runtime input tensors for the model.

    - x: Tensor of shape (BATCH_SIZE, CHANNELS, DEPTH, HEIGHT, WIDTH)
    """
    x = torch.randn(BATCH_SIZE, CHANNELS, DEPTH, HEIGHT, WIDTH)
    return [x]

def get_init_inputs() -> List:
    """
    Returns the initialization parameters for the Model constructor in the same order:
    [channels, pool_kernel, pool_stride, pad, pad_value, dropout_p]
    """
    return [CHANNELS, POOL_KERNEL, POOL_STRIDE, PAD, PAD_VALUE, DROPOUT_P]