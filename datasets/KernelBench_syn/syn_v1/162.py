import torch
import torch.nn as nn
from typing import List, Any, Tuple

class Model(nn.Module):
    """
    A moderately complex 3D vision-style module that demonstrates:
      - Lazy convolution initialization (nn.LazyConv3d)
      - Two stages of fractional max pooling (nn.FractionalMaxPool3d)
      - Channel-wise dropout applied to flattened spatial dimensions (nn.Dropout1d)
      - A final projection (nn.Linear) after spatial-global aggregation

    Forward computation summary:
      1) 3D convolution (LazyConv3d) -> ReLU
      2) FractionalMaxPool3d stage 1
      3) ReLU
      4) FractionalMaxPool3d stage 2
      5) Reshape from (N, C, D, H, W) to (N, C, L) where L = D*H*W
      6) Dropout1d applied on (N, C, L) zeroing entire channels across spatial length
      7) Global spatial mean (N, C)
      8) Final linear projection (N, proj_dim)
    """
    def __init__(
        self,
        out_channels: int,
        conv_kernel_size: Tuple[int, int, int],
        conv_stride: Tuple[int, int, int] = (1, 1, 1),
        conv_padding: Tuple[int, int, int] = (0, 0, 0),
        pool_kernel: Tuple[int, int, int] = (2, 2, 2),
        pool_ratio1: Tuple[float, float, float] = (0.6, 0.6, 0.6),
        pool_ratio2: Tuple[float, float, float] = (0.7, 0.7, 0.7),
        dropout_p: float = 0.2,
        proj_dim: int = 128
    ):
        super(Model, self).__init__()
        # LazyConv3d will infer in_channels at first forward call
        self.conv = nn.LazyConv3d(out_channels, kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding)
        # Fractional pooling stages - note these accept kernel_size and an output_ratio tuple
        self.pool1 = nn.FractionalMaxPool3d(kernel_size=pool_kernel, output_ratio=pool_ratio1)
        self.pool2 = nn.FractionalMaxPool3d(kernel_size=pool_kernel, output_ratio=pool_ratio2)
        # Channel-wise dropout over the flattened spatial dimension: expects input (N, C, L)
        self.dropout = nn.Dropout1d(p=dropout_p)
        # Final projection (after global spatial aggregation)
        self.proj = nn.Linear(out_channels, proj_dim)
        # Small activation
        self.act = nn.ReLU(inplace=True)

    def _maybe_unpack_fractional(self, out):
        """
        FractionalMaxPool3d may return (output, indices). Ensure we return just the tensor.
        """
        if isinstance(out, tuple) or isinstance(out, list):
            return out[0]
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor with shape (N, C_in, D, H, W)

        Returns:
            Tensor of shape (N, proj_dim)
        """
        # 1) Convolution + ReLU
        x = self.conv(x)
        x = self.act(x)

        # 2) Fractional max pooling stage 1
        out = self.pool1(x)
        x = self._maybe_unpack_fractional(out)

        # 3) Non-linearity
        x = self.act(x)

        # 4) Fractional max pooling stage 2
        out = self.pool2(x)
        x = self._maybe_unpack_fractional(out)

        # 5) Reshape spatial dims into a single length dimension for Dropout1d
        N, C, D, H, W = x.shape
        L = D * H * W
        x = x.view(N, C, L)  # (N, C, L)

        # 6) Channel-wise dropout (zeros whole channels across the spatial length)
        x = self.dropout(x)

        # 7) Global spatial aggregation (mean over length)
        x = x.mean(dim=2)  # (N, C)

        # 8) Final projection to desired output dimension
        x = self.proj(x)  # (N, proj_dim)
        return x

# Configuration / default sizes for inputs and initialization
batch_size = 8
in_channels = 3  # will be inferred by LazyConv3d, but we still provide realistic input
depth = 32
height = 64
width = 64

# Initialization parameters for the model
out_channels = 32
conv_kernel_size = (3, 3, 3)
conv_stride = (1, 1, 1)
conv_padding = (1, 1, 1)
pool_kernel = (2, 2, 2)
pool_ratio1 = (0.6, 0.6, 0.5)
pool_ratio2 = (0.7, 0.5, 0.6)
dropout_p = 0.25
proj_dim = 256

def get_inputs() -> List[torch.Tensor]:
    """
    Returns a list containing a single 5D input tensor with shape (batch_size, in_channels, depth, height, width).
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs() -> List[Any]:
    """
    Returns the initialization parameters needed to construct the Model in the order:
      [out_channels, conv_kernel_size, conv_stride, conv_padding, pool_kernel, pool_ratio1, pool_ratio2, dropout_p, proj_dim]
    """
    return [out_channels, conv_kernel_size, conv_stride, conv_padding, pool_kernel, pool_ratio1, pool_ratio2, dropout_p, proj_dim]