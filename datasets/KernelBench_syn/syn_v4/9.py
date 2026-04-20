import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex model that combines 2D batch normalization, a reshape that creates a 3D depth
    dimension from channel groups, Lp pooling in 3D, and a lazy 3D batch normalization.
    The pipeline is:
      1. BatchNorm2d over (N, C, H, W)
      2. ReLU activation
      3. Reshape channels -> (C_per_slice, D) to form (N, C_per_slice, D, H, W)
      4. LpPool3d over the (D, H, W) spatial dims
      5. LazyBatchNorm3d (lazily initialized on first forward)
      6. AdaptiveAvgPool3d to (1,1,1) and L2-normalize the resulting feature vector
    """
    def __init__(self, depth_slices: int = 3, lp_norm: int = 2, lp_kernel=(2, 2, 2)):
        """
        Args:
            depth_slices (int): Number of depth slices to carve out from channels
                                (channels must be divisible by this).
            lp_norm (int): The norm degree for LPPool3d (p-norm).
            lp_kernel (tuple): Kernel size for LPPool3d as (D_k, H_k, W_k).
        """
        super(Model, self).__init__()
        self.depth_slices = depth_slices
        self.lp_norm = lp_norm
        self.lp_kernel = lp_kernel

        # 2D batch norm operates on the full input channels (set as global CHANNELS)
        # This will be instantiated at construction time because CHANNELS is known.
        self.bn2d = nn.BatchNorm2d(CHANNELS)

        # Lp pooling in 3D over (depth, height, width)
        self.lppool3d = nn.LPPool3d(norm_type=self.lp_norm, kernel_size=self.lp_kernel)

        # LazyBatchNorm3d will determine its num_features on first forward pass.
        # It's useful here because after reshaping channels -> C_per_slice we may not want
        # to hard-code that number in the module definition.
        self.bn3d = nn.LazyBatchNorm3d()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor with shape (N, C, H, W).

        Returns:
            torch.Tensor: Output tensor with shape (N, C_per_slice) where C_per_slice = C // depth_slices
                          (L2-normalized per sample).
        """
        # 1) Normalize across channels for 2D spatial input
        x = self.bn2d(x)

        # 2) Simple non-linearity
        x = F.relu(x)

        # 3) Form a depth dimension by splitting channels into 'depth_slices' groups
        N, C, H, W = x.shape
        assert C % self.depth_slices == 0, "Input channels must be divisible by depth_slices"
        C_per_slice = C // self.depth_slices

        # Reshape from (N, C, H, W) -> (N, C_per_slice, D, H, W)
        x = x.view(N, C_per_slice, self.depth_slices, H, W)

        # 4) Apply 3D Lp pooling over (D, H, W)
        x = self.lppool3d(x)

        # 5) Apply lazy 3D batch normalization (initializes on first forward)
        x = self.bn3d(x)

        # 6) Global pooling to a compact feature and normalize
        x = F.adaptive_avg_pool3d(x, output_size=(1, 1, 1))  # (N, C_per_slice, 1, 1, 1)
        x = x.view(N, C_per_slice)  # (N, C_per_slice)
        x = F.normalize(x, p=2, dim=1)

        return x

# Configuration variables
BATCH_SIZE = 8
CHANNELS = 12  # Must be divisible by DEPTH_SLICES
HEIGHT = 32
WIDTH = 32
DEPTH_SLICES = 3  # Number of depth slices to carve out from channels

def get_inputs():
    """
    Returns a list containing a single input tensor appropriate for the Model.
    Shape: (BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)
    """
    x = torch.randn(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for the Model constructor.
    Here we provide the depth_slices value to ensure deterministic behavior.
    """
    return [DEPTH_SLICES]