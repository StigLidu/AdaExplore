import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex 3D -> 2D fusion module that:
    - Applies 3D max-pooling to reduce spatial resolution.
    - Folds the reduced depth into the channels dimension to perform 2D batch norm.
    - Applies Local Response Normalization and a ReLU non-linearity.
    - Restores the original channel/depth structure and reduces over depth.

    Input shape: (B, C, D, H, W)
    Output shape: (B, C, 1, H_out, W_out) where H_out and W_out are reduced by the pooling.
    """
    def __init__(self,
                 in_channels: int,
                 depth: int,
                 pool_kernel: tuple = (2, 2, 2),
                 bn_momentum: float = 0.1,
                 lrn_size: int = 5):
        """
        Args:
            in_channels (int): Number of input channels C.
            depth (int): Depth dimension size D (must be divisible by pool_kernel[0]).
            pool_kernel (tuple): 3-tuple for MaxPool3d kernel and stride.
            bn_momentum (float): Momentum value for BatchNorm2d.
            lrn_size (int): Size parameter for LocalResponseNorm.
        """
        super(Model, self).__init__()
        assert len(pool_kernel) == 3, "pool_kernel must be a 3-tuple"
        kd = pool_kernel[0]
        assert depth % kd == 0, "depth must be divisible by pool_kernel[0]"

        # 3D pooling to reduce (D, H, W)
        self.pool3d = nn.MaxPool3d(kernel_size=pool_kernel, stride=pool_kernel)

        # After pooling, we will fold the depth dimension into channels:
        pooled_depth = depth // kd
        bn_num_features = in_channels * pooled_depth

        # 2D BatchNorm on the folded tensor
        self.bn2d = nn.BatchNorm2d(bn_num_features, momentum=bn_momentum)

        # Local Response Normalization applied after BatchNorm
        self.lrn = nn.LocalResponseNorm(lrn_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Steps:
        1. 3D max pool: (B, C, D, H, W) -> (B, C, D1, H1, W1)
        2. Permute and reshape to fold depth into channels:
           (B, C, D1, H1, W1) -> (B, C*D1, H1, W1)
        3. Apply BatchNorm2d -> LocalResponseNorm -> ReLU
        4. Restore original channel/depth layout.
        5. Reduce (mean) over depth dimension to produce (B, C, 1, H1, W1)
        """
        # 1. 3D pooling
        p = self.pool3d(x)  # (B, C, D1, H1, W1)

        B, C, D1, H1, W1 = p.shape

        # 2. Fold depth into channels for 2D operations
        # Permute to (B, D1, C, H1, W1) then combine D1 and C -> (B, C*D1, H1, W1)
        p_perm = p.permute(0, 2, 1, 3, 4)  # (B, D1, C, H1, W1)
        p_fold = p_perm.reshape(B, C * D1, H1, W1)  # (B, C*D1, H1, W1)

        # 3. Normalize and non-linearity
        p_bn = self.bn2d(p_fold)
        p_lrn = self.lrn(p_bn)
        p_act = torch.relu(p_lrn)

        # 4. Restore to (B, C, D1, H1, W1)
        p_unfold = p_act.reshape(B, D1, C, H1, W1).permute(0, 2, 1, 3, 4)

        # 5. Reduce over depth (D1) to produce output with depth dimension of 1
        out = torch.mean(p_unfold, dim=2, keepdim=True)  # (B, C, 1, H1, W1)

        return out

# Configuration variables
batch_size = 8
in_channels = 3
depth = 16  # must be divisible by pool_kernel[0]
height = 64
width = 64
pool_kernel = (2, 2, 2)
bn_momentum = 0.1
lrn_size = 5

def get_inputs():
    """
    Returns a list containing a single input tensor with shape:
    (batch_size, in_channels, depth, height, width)
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization inputs for the Model constructor in order:
    [in_channels, depth, pool_kernel, bn_momentum, lrn_size]
    """
    return [in_channels, depth, pool_kernel, bn_momentum, lrn_size]