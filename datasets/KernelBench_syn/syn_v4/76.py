import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class Model(nn.Module):
    """
    Complex example combining ZeroPad3d, MaxPool2d/MaxUnpool2d and RReLU in a non-trivial dataflow.

    Steps:
    1. Expand a 4D image tensor (N, C, H, W) into 5D by adding a depth dimension.
    2. Apply ZeroPad3d to pad depth, height and width.
    3. Collapse the depth dimension into channels to obtain a 4D tensor suitable for 2D pooling.
    4. Apply MaxPool2d (with indices) and then MaxUnpool2d to reverse the pooling (using indices).
    5. Restore the 5D layout, reduce the depth dimension by averaging, crop away padding to original spatial size.
    6. Apply RReLU activation as a final non-linearity.
    """
    def __init__(
        self,
        pad: Tuple[int, int, int, int, int, int] = (1, 1, 2, 2, 1, 1),
        pool_kernel: Tuple[int, int] = (2, 2),
        pool_stride: Tuple[int, int] = (2, 2),
        rrelu_lower: float = 0.125,
        rrelu_upper: float = 0.333,
    ):
        """
        Initializes the modules used in forward.

        Args:
            pad: 6-tuple for ZeroPad3d (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back).
            pool_kernel: kernel size for MaxPool2d / MaxUnpool2d.
            pool_stride: stride for pooling / unpooling.
            rrelu_lower: lower bound for randomized slope in RReLU.
            rrelu_upper: upper bound for randomized slope in RReLU.
        """
        super(Model, self).__init__()
        self.pad = pad
        self.pool_kernel = pool_kernel
        self.pool_stride = pool_stride

        # ZeroPad3d operates on 5D tensors: (N, C, D, H, W)
        self.pad3d = nn.ZeroPad3d(pad)

        # MaxPool2d with return_indices to be used for MaxUnpool2d
        self.pool2d = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride, return_indices=True)
        self.unpool2d = nn.MaxUnpool2d(kernel_size=pool_kernel, stride=pool_stride)

        # Randomized leaky ReLU
        self.rrelu = nn.RReLU(lower=rrelu_lower, upper=rrelu_upper, inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining padding, pooling/unpooling and randomized activation.

        Args:
            x: Input tensor of shape (N, C, H, W)

        Returns:
            Tensor of shape (N, C, H, W) after the composed operations.
        """
        if x.dim() != 4:
            raise ValueError("Input must be a 4D tensor (N, C, H, W)")

        N, C, H_orig, W_orig = x.shape
        pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back = self.pad

        # 1) Add a depth dimension to make it 5D: (N, C, D=1, H, W)
        x5 = x.unsqueeze(2)  # shape: (N, C, 1, H, W)

        # 2) Zero-pad in 3D (depth, height, width padded as specified)
        x_padded = self.pad3d(x5)  # shape: (N, C, D_p, H_p, W_p)
        N, C, D_p, H_p, W_p = x_padded.shape

        # 3) Collapse depth into channels to perform 2D pooling: new_channels = C * D_p
        x_2d = x_padded.view(N, C * D_p, H_p, W_p)  # shape: (N, C*D_p, H_p, W_p)

        # 4) MaxPool2d with indices, then MaxUnpool2d to reverse the pooling (using the stored indices)
        pooled, indices = self.pool2d(x_2d)  # pooled shape smaller
        # Use the original 2D shape as output_size for unpool to ensure deterministic size
        unpooled = self.unpool2d(pooled, indices, output_size=x_2d.size())  # shape: (N, C*D_p, H_p, W_p)

        # 5) Restore 5D layout: (N, C, D_p, H_p, W_p)
        restored5 = unpooled.view(N, C, D_p, H_p, W_p)

        # Reduce the depth dimension by taking the mean across depth -- alternative to slicing
        reduced = restored5.mean(dim=2)  # shape: (N, C, H_p, W_p)

        # 6) Crop away the padding to return to original spatial dimensions (H_orig, W_orig)
        h_start = pad_top
        h_end = h_start + H_orig
        w_start = pad_left
        w_end = w_start + W_orig

        cropped = reduced[:, :, h_start:h_end, w_start:w_end]

        # 7) Apply randomized leaky ReLU activation
        out = self.rrelu(cropped)

        return out


# Configuration variables
BATCH_SIZE = 4
CHANNELS = 3
HEIGHT = 64
WIDTH = 64

# Padding: (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
PAD = (1, 1, 2, 2, 1, 1)

# Pooling parameters
POOL_KERNEL = (2, 2)
POOL_STRIDE = (2, 2)

def get_inputs() -> List[torch.Tensor]:
    """
    Generates a random input tensor matching the configured batch, channels and spatial dimensions.

    Returns:
        List containing the input tensor.
    """
    x = torch.randn(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)
    return [x]

def get_init_inputs() -> List:
    """
    Returns the initialization parameters for the Model constructor.

    Returns:
        List containing the pad tuple, pool kernel and pool stride.
    """
    return [PAD, POOL_KERNEL, POOL_STRIDE]