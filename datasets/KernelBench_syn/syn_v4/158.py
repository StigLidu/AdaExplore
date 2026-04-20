import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class Model(nn.Module):
    """
    Complex patch-transform model that:
    - Applies ReLU activation
    - Performs 2D max pooling to reduce spatial resolution
    - Extracts overlapping patches using unfold
    - Applies a learned linear transform to each patch (im2col style)
    - Reconstructs the spatial tensor using nn.Fold
    - Normalizes overlaps (averaging) and applies a final ReLU

    This model demonstrates a combination of nn.MaxPool2d, nn.ReLU and nn.Fold,
    and uses matrix multiplication on unfolded patches to implement a patch-wise
    linear operator that is then folded back to image space.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        pool_kernel: int,
        input_height: int,
        input_width: int,
        eps: float = 1e-6
    ):
        """
        Initialize the model.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels after patch transform.
            kernel_size (int): Spatial kernel size for patches.
            stride (int): Stride for extracting patches (affects overlap).
            pool_kernel (int): Kernel size / stride for max pooling.
            input_height (int): Height of the input images.
            input_width (int): Width of the input images.
            eps (float, optional): Small value to avoid division by zero when normalizing. Defaults to 1e-6.
        """
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pool_kernel = pool_kernel
        self.input_height = input_height
        self.input_width = input_width
        self.eps = eps

        # Basic layers
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=self.pool_kernel, stride=self.pool_kernel)

        # Compute pooled spatial size to configure Fold
        pooled_h = self.input_height // self.pool_kernel
        pooled_w = self.input_width // self.pool_kernel
        self.output_size = (pooled_h, pooled_w)

        # nn.Fold expects input channel dimension = out_channels * (kernel_size**2)
        self.fold = nn.Fold(output_size=self.output_size, kernel_size=self.kernel_size, stride=self.stride)

        # Linear mapping parameters for patches:
        # Each patch is a vector of length in_channels * kernel_size * kernel_size
        self.patch_dim_in = self.in_channels * (self.kernel_size ** 2)
        # We map each patch to a vector of length out_channels * kernel_size * kernel_size
        self.patch_dim_out = self.out_channels * (self.kernel_size ** 2)

        # Learned weight and bias for the patch transform
        # Initialized small to stabilize training in general use
        self.weight = nn.Parameter(torch.randn(self.patch_dim_out, self.patch_dim_in) * 0.02)
        self.bias = nn.Parameter(torch.zeros(self.patch_dim_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_channels, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch, out_channels, pooled_h, pooled_w).
        """
        batch = x.shape[0]

        # 1) Pointwise non-linearity
        x = self.relu(x)

        # 2) Spatial reduction via max-pooling
        pooled = self.pool(x)  # shape: (batch, in_channels, pooled_h, pooled_w)

        # 3) Extract overlapping patches (im2col)
        # patches: (batch, in_channels * kernel_size * kernel_size, L)
        patches = F.unfold(pooled, kernel_size=self.kernel_size, stride=self.stride)

        # Ensure the patch dimensionality matches expectations
        assert patches.shape[1] == self.patch_dim_in, \
            f"Expected unfolded patch dim {self.patch_dim_in}, got {patches.shape[1]}"

        # 4) Apply learned linear transform to each patch:
        # Broadcast weight to batch and perform batch matrix multiply
        # weight_expanded: (batch, patch_dim_out, patch_dim_in)
        weight_expanded = self.weight.unsqueeze(0).expand(batch, -1, -1)
        # transformed: (batch, patch_dim_out, L)
        transformed = torch.bmm(weight_expanded, patches)
        # Add bias: (patch_dim_out,) -> (batch, patch_dim_out, 1) -> broadcast across L
        transformed = transformed + self.bias.unsqueeze(0).unsqueeze(-1)

        # 5) Fold back to spatial domain
        # fold expects input shape (batch, out_channels * k*k, L)
        out = self.fold(transformed)  # shape: (batch, out_channels, pooled_h, pooled_w)

        # 6) Normalize by overlap count: compute fold of ones to get how many times each pixel was summed
        ones = torch.ones_like(transformed)
        overlap = self.fold(ones)  # same shape as out: (batch, out_channels, pooled_h, pooled_w)
        out = out / (overlap + self.eps)

        # 7) Final non-linearity
        out = self.relu(out)

        return out

# Configuration variables (module-level)
batch_size = 8
in_channels = 3
out_channels = 6
input_height = 32
input_width = 32
kernel_size = 3
stride = 1
pool_kernel = 2

def get_inputs() -> List[torch.Tensor]:
    """
    Create a list containing a single input tensor for the model.

    Returns:
        List[torch.Tensor]: [x] where x has shape (batch_size, in_channels, input_height, input_width)
    """
    x = torch.randn(batch_size, in_channels, input_height, input_width)
    return [x]

def get_init_inputs() -> List:
    """
    Provide initialization parameters for Model constructor in the expected order.

    Returns:
        List: [in_channels, out_channels, kernel_size, stride, pool_kernel, input_height, input_width]
    """
    return [in_channels, out_channels, kernel_size, stride, pool_kernel, input_height, input_width]