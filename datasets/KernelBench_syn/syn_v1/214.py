import torch
import torch.nn as nn
from typing import Tuple, List, Any

class Model(nn.Module):
    """
    Complex 3D feature extractor that demonstrates a pipeline combining:
      - Constant 3D padding
      - Lazy 3D convolution (in_channels inferred at first forward)
      - ReLU non-linearity
      - Spatial flattening (D,H,W -> sequence)
      - 1D average pooling over the flattened spatial sequence
      - Per-channel L2 normalization across the pooled sequence
      - Global aggregation and a final linear projection

    The model returns a compact per-sample feature vector of size `out_channels`.
    """
    def __init__(
        self,
        out_channels: int,
        kernel_size: Tuple[int, int, int],
        pool_kernel: int = 4,
        pad_val: float = 0.0,
        pad: Tuple[int, int, int, int, int, int] = (1,1,1,1,1,1)
    ):
        """
        Args:
            out_channels (int): Number of output channels for the Conv3d.
            kernel_size (tuple): 3D convolution kernel size (kd, kh, kw).
            pool_kernel (int): Kernel size (and stride) for AvgPool1d applied to flattened spatial dims.
            pad_val (float): Constant value used when padding with ConstantPad3d.
            pad (tuple): 6-tuple padding for ConstantPad3d (left, right, top, bottom, front, back).
        """
        super(Model, self).__init__()
        # Padding layer applied before convolution
        self.pad3d = nn.ConstantPad3d(padding=pad, value=pad_val)
        # LazyConv3d will infer in_channels from the first input it sees
        self.conv3d = nn.LazyConv3d(out_channels=out_channels, kernel_size=kernel_size, stride=1)
        self.relu = nn.ReLU(inplace=True)
        # AvgPool1d to reduce the flattened spatial sequence length
        self.pool1d = nn.AvgPool1d(kernel_size=pool_kernel, stride=pool_kernel)
        # Final linear projection operates on per-channel aggregated features
        self.fc = nn.Linear(out_channels, out_channels)
        # small epsilon to stabilize division in normalization
        self.eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the multi-stage pipeline.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C_in, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (N, out_channels) - compact per-sample features.
        """
        # Step 1: Constant padding to control spatial dimensions before convolution
        x_padded = self.pad3d(x)                           # (N, C_in, D+pad, H+pad, W+pad)

        # Step 2: 3D convolution (LazyConv3d infers C_in at first pass)
        conv_out = self.conv3d(x_padded)                   # (N, out_channels, D_out, H_out, W_out)

        # Step 3: Non-linearity
        activated = self.relu(conv_out)                    # (N, out_channels, D_out, H_out, W_out)

        # Step 4: Flatten spatial dims (D_out, H_out, W_out) -> sequence length L
        n, c, d, h, w = activated.shape
        seq = activated.view(n, c, d * h * w)              # (N, out_channels, L)

        # Step 5: 1D average pooling over the flattened spatial sequence to reduce its length
        pooled = self.pool1d(seq)                          # (N, out_channels, L')

        # Step 6: Per-channel L2 normalization across the pooled sequence length
        # Compute L2 norm across the last dim and normalize
        norms = pooled.norm(p=2, dim=2, keepdim=True)     # (N, out_channels, 1)
        normalized = pooled / (norms + self.eps)          # (N, out_channels, L')

        # Step 7: Global aggregation: average across the reduced sequence to get a channel vector
        global_feat = normalized.mean(dim=2)               # (N, out_channels)

        # Step 8: Final linear projection to produce output features
        out = self.fc(global_feat)                         # (N, out_channels)

        return out

# Module-level configuration (input sizes and initialization parameters)
batch_size = 4
in_channels = 3          # Will be implicitly inferred by LazyConv3d during the first forward
depth = 8
height = 8
width = 8

out_channels = 32
conv_kernel = (3, 3, 3)
pool_kernel = 4
pad_value = 0.1
pad_tuple = (1, 1, 1, 1, 1, 1)  # left, right, top, bottom, front, back

def get_inputs() -> List[torch.Tensor]:
    """
    Returns a list containing the input tensor for the model.
    Input shape: (batch_size, in_channels, depth, height, width)
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs() -> List[Any]:
    """
    Returns initialization parameters for the Model:
      [out_channels, conv_kernel, pool_kernel, pad_value, pad_tuple]
    """
    return [out_channels, conv_kernel, pool_kernel, pad_value, pad_tuple]