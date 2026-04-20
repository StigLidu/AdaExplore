import torch
import torch.nn as nn
from typing import Tuple, List

class Model(nn.Module):
    """
    Complex module combining AdaptiveAvgPool3d, ReLU, and AvgPool1d with
    a small post-pooling channel-wise normalization.

    Computation pipeline:
      1. AdaptiveAvgPool3d reduces spatial dimensions to a target output size.
      2. ReLU activation is applied element-wise.
      3. The pooled 3D spatial dimensions are flattened into a 1D sequence per channel.
      4. AvgPool1d is applied across that sequence to aggregate local temporal information.
      5. Channel-wise normalization (mean/std) is performed across the resulting sequence.

    This produces a tensor of shape (batch_size, channels, pooled_length_after_avgpool1d).
    """
    def __init__(self, adaptive_out: Tuple[int, int, int], pool1d_kernel: int = 3, pool1d_stride: int = 1, eps: float = 1e-6):
        """
        Args:
            adaptive_out (tuple): Output size for AdaptiveAvgPool3d (out_d, out_h, out_w).
            pool1d_kernel (int): Kernel size for AvgPool1d applied after flattening spatial dims.
            pool1d_stride (int): Stride for AvgPool1d.
            eps (float): Small epsilon for normalization to avoid division by zero.
        """
        super(Model, self).__init__()
        self.adaptive_pool = nn.AdaptiveAvgPool3d(adaptive_out)
        self.relu = nn.ReLU(inplace=False)
        self.avgpool1d = nn.AvgPool1d(kernel_size=pool1d_kernel, stride=pool1d_stride)
        self.eps = eps
        # store adaptive output for reshaping logic
        self.adaptive_out = adaptive_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the composed operations.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, depth, height, width)

        Returns:
            torch.Tensor: Tensor of shape (batch_size, channels, L_out) where L_out is the
                          length after flattening the adaptive pool output and applying AvgPool1d.
        """
        # 1) Adaptive average pooling to fixed spatial size
        pooled3d = self.adaptive_pool(x)  # (B, C, out_d, out_h, out_w)

        # 2) Non-linearity
        activated = self.relu(pooled3d)  # (B, C, out_d, out_h, out_w)

        # 3) Flatten spatial dims into a single sequence per channel
        B, C, out_d, out_h, out_w = activated.shape
        seq_len = out_d * out_h * out_w
        # reshape to (B, C, seq_len) for AvgPool1d
        seq = activated.view(B, C, seq_len)

        # 4) 1D average pooling across the sequence dimension
        pooled1d = self.avgpool1d(seq)  # (B, C, L_out)

        # 5) Channel-wise normalization across the sequence dimension
        mean = pooled1d.mean(dim=2, keepdim=True)  # (B, C, 1)
        std = pooled1d.std(dim=2, unbiased=False, keepdim=True)  # (B, C, 1)
        normalized = (pooled1d - mean) / (std + self.eps)

        return normalized


# Configuration / default inputs
batch_size = 8
channels = 16
depth = 20
height = 18
width = 18

# Adaptive output designed so that flattened length is >= pool1d_kernel
adaptive_out_d = 4
adaptive_out_h = 2
adaptive_out_w = 2

# Derived flattened length = 4 * 2 * 2 = 16
# Choose AvgPool1d kernel and stride that make sense for length=16
pool1d_kernel = 3
pool1d_stride = 2

adaptive_out = (adaptive_out_d, adaptive_out_h, adaptive_out_w)

def get_inputs():
    """
    Returns:
        List containing a single input tensor of shape (batch_size, channels, depth, height, width).
    """
    x = torch.randn(batch_size, channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Returns:
        Initialization parameters for the Model: [adaptive_out, pool1d_kernel, pool1d_stride]
    """
    return [adaptive_out, pool1d_kernel, pool1d_stride]