import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class Model(nn.Module):
    """
    A 1D signal processing module that demonstrates a mixed pattern of padding,
    power-average pooling, and a transposed convolution applied in a 2D fashion
    to perform channel and length transformations.

    Computation graph (high-level):
      1. ReplicationPad1d on the temporal axis
      2. LPPool1d to perform a p-norm pooling (reduces temporal resolution)
      3. Unsqueeze to convert to 4D (N, C, H=1, W=time) for ConvTranspose2d
      4. LazyConvTranspose2d to increase channels and upsample along time
      5. Squeeze back to 3D and add a projected residual path (1x1 Conv1d)
      6. Interpolate residual to match output length, elementwise add and ReLU
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_p: int = 2,
        pool_kernel: int = 3,
        pool_stride: int = 2,
        deconv_kernel: tuple = (1, 4),
        deconv_stride: tuple = (1, 2),
        pad: tuple = (2, 1),
    ):
        """
        Args:
            in_channels (int): Number of input channels for the 1D signal.
            out_channels (int): Desired output channels after the transposed convolution.
            pool_p (int): The "p" norm for LPPool1d (e.g., 2 for L2 pooling).
            pool_kernel (int): Kernel size for LPPool1d.
            pool_stride (int): Stride for LPPool1d.
            deconv_kernel (tuple): Kernel size for ConvTranspose2d (H, W). H should be 1.
            deconv_stride (tuple): Stride for ConvTranspose2d (H, W).
            pad (tuple): ReplicationPad1d padding (left, right).
        """
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pool_p = pool_p
        self.pool_kernel = pool_kernel
        self.pool_stride = pool_stride
        self.deconv_kernel = deconv_kernel
        self.deconv_stride = deconv_stride
        self.pad = pad

        # Replication padding on the temporal (last) dimension
        self.rep_pad = nn.ReplicationPad1d(self.pad)

        # LPPool1d performs a p-norm based pooling (reduces temporal resolution)
        self.pool = nn.LPPool1d(norm_type=self.pool_p, kernel_size=self.pool_kernel, stride=self.pool_stride)

        # LazyConvTranspose2d will infer the in_channels upon the first forward pass.
        # We use a (1, K) kernel so the operation only affects the temporal dimension.
        self.deconv = nn.LazyConvTranspose2d(
            out_channels=self.out_channels,
            kernel_size=self.deconv_kernel,
            stride=self.deconv_stride,
            padding=(0, 1)  # small spatial padding; H-pad=0 keeps the singleton dim intact
        )

        # A projection to map the pooled input channels to the deconv output channels
        # so we can form a channel-wise residual connection. This uses the known in_channels.
        self.project = nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, new_length)
        """
        # 1. Replicate pad the 1D temporal axis
        x_padded = self.rep_pad(x)

        # 2. LPPool1d (p-norm pooling). This reduces the temporal resolution.
        x_pooled = self.pool(x_padded)  # shape: (N, C, L_pool)

        # 3. Convert to 4D for ConvTranspose2d by adding a singleton spatial dim
        x_4d = x_pooled.unsqueeze(2)  # shape: (N, C, 1, L_pool)

        # 4. Apply LazyConvTranspose2d.
        #    This will infer its in_channels from x_4d and produce (N, out_channels, 1, L_out)
        y_4d = self.deconv(x_4d)

        # 5. Squeeze back to 3D (temporal signal)
        y = y_4d.squeeze(2)  # shape: (N, out_channels, L_out)

        # 6. Residual path: project the pooled input channels to match out_channels
        res = self.project(x_pooled)  # shape: (N, out_channels, L_pool)

        # 7. Upsample the residual to match the deconv output length (linear interpolation)
        if res.size(-1) != y.size(-1):
            res_upsampled = F.interpolate(res, size=y.size(-1), mode='linear', align_corners=False)
        else:
            res_upsampled = res

        # 8. Combine and apply non-linearity
        out = F.relu(y + res_upsampled)

        return out

# Configuration variables for the test harness
batch_size = 8
in_channels = 12
out_channels = 20
input_length = 1024

# Default layer hyperparameters (exposed for potential test initialization)
pool_p = 2
pool_kernel = 3
pool_stride = 2
deconv_kernel = (1, 4)
deconv_stride = (1, 2)
pad = (2, 1)

def get_inputs() -> List[torch.Tensor]:
    """
    Returns a list with a single input tensor for the Model forward pass.
    Shape: (batch_size, in_channels, input_length)
    """
    x = torch.randn(batch_size, in_channels, input_length)
    return [x]

def get_init_inputs() -> List:
    """
    Returns the initialization parameters required to construct the Model.
    """
    return [in_channels, out_channels, pool_p, pool_kernel, pool_stride, deconv_kernel, deconv_stride, pad]