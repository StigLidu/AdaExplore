import torch
import torch.nn as nn
from typing import List

class Model(nn.Module):
    """
    Complex patch-based reconstruction model that:
    - Pads the input spatially with a constant value (ConstantPad2d)
    - Extracts sliding local blocks (Unfold)
    - Applies a Log-Sigmoid nonlinearity to each patch element (LogSigmoid)
    - Scales patch elements by a learned per-element factor
    - Reconstructs the spatial tensor from modified patches using Fold (Fold)
    - Normalizes overlapped contributions and crops to the original spatial size
    - Produces a compact per-channel descriptor by spatial averaging

    The model demonstrates a non-trivial pipeline combining padding, patch extraction,
    elementwise nonlinearities, learnable scaling, and patch reassembly.
    """
    def __init__(
        self,
        in_channels: int,
        height: int,
        width: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        pad_value: float = 0.0,
    ):
        """
        Initialize the model.

        Args:
            in_channels (int): Number of input channels.
            height (int): Height of the input images.
            width (int): Width of the input images.
            kernel_size (int): Size of the sliding patch (assumed square).
            stride (int): Stride between patches.
            padding (int): Symmetric padding to apply on all sides before patch extraction.
            dilation (int): Dilation applied to the patch extraction.
            pad_value (float): Constant value used to pad image boundaries.
        """
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.pad_value = pad_value

        # Padding layer (ConstantPad2d)
        # ConstantPad2d takes padding as single int or tuple (left,right,top,bottom).
        self.pad = nn.ConstantPad2d(self.padding, self.pad_value)

        # Unfold / Fold parameters:
        # We extract patches from the padded tensor, then re-assemble them to the padded size.
        padded_h = self.height + 2 * self.padding
        padded_w = self.width + 2 * self.padding
        self.padded_size = (padded_h, padded_w)

        # Unfold is used to extract sliding local blocks (not stored as module, created here)
        # Fold reconstructs the tensor from modified patches.
        self.unfold = nn.Unfold(
            kernel_size=(self.kernel_size, self.kernel_size),
            dilation=self.dilation,
            padding=0,  # padding already applied by ConstantPad2d
            stride=self.stride
        )
        self.fold = nn.Fold(
            output_size=self.padded_size,
            kernel_size=(self.kernel_size, self.kernel_size),
            dilation=self.dilation,
            padding=0,
            stride=self.stride
        )

        # Log-sigmoid nonlinearity applied to patch elements
        self.logsigmoid = nn.LogSigmoid()

        # Per-patch-element learnable scaling vector (applied channel-wise within the patch)
        # Shape: (in_channels * kernel_size * kernel_size,)
        scale_shape = self.in_channels * (self.kernel_size * self.kernel_size)
        self.scale = nn.Parameter(torch.ones(scale_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the pipeline.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (B, C) representing spatially averaged
                          per-channel descriptors after patch-based processing.
        """
        # 1) Pad spatial boundaries with a constant
        x_padded = self.pad(x)  # shape: (B, C, H + 2*pad, W + 2*pad)

        # 2) Extract sliding patches: shape (B, C * k * k, L)
        patches = self.unfold(x_padded)

        # 3) Apply LogSigmoid nonlinearity element-wise
        patches = self.logsigmoid(patches)

        # 4) Scale each patch element by a learned factor (broadcast across batch and positions)
        # scale shape -> (1, C*k*k, 1) to match (B, C*k*k, L)
        patches = patches * self.scale.view(1, -1, 1)

        # 5) Reconstruct the padded spatial tensor from modified patches using Fold
        reconstructed = self.fold(patches)  # shape: (B, C, padded_h, padded_w)

        # 6) Normalize overlaps: compute fold of ones to get contribution counts and divide
        ones = torch.ones_like(patches)
        overlap_count = self.fold(ones)
        # Prevent division by zero; overlap_count > 0 where valid
        reconstructed = reconstructed / (overlap_count + 1e-6)

        # 7) Crop back to original spatial dimensions
        if self.padding > 0:
            reconstructed = reconstructed[
                :, :, self.padding : self.padding + self.height, self.padding : self.padding + self.width
            ]
        # 8) Final spatial aggregation: mean over H and W to produce per-channel descriptors
        out = reconstructed.mean(dim=(2, 3))  # shape: (B, C)
        return out

# Configuration variables
batch_size = 8
channels = 3
height = 64
width = 64
kernel_size = 5
stride = 2
padding = 2
dilation = 1
pad_value = 0.1

def get_inputs() -> List[torch.Tensor]:
    """
    Creates and returns a list containing a single input tensor for the model.

    The tensor has shape (batch_size, channels, height, width).
    """
    x = torch.randn(batch_size, channels, height, width)
    return [x]

def get_init_inputs() -> List:
    """
    Returns the list of initialization arguments expected by Model.__init__:
    [in_channels, height, width, kernel_size, stride, padding, dilation, pad_value]
    """
    return [channels, height, width, kernel_size, stride, padding, dilation, pad_value]