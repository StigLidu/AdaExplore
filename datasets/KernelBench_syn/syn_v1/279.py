import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex model that combines 3D batch normalization, spatial pooling across
    merged channel-depth dimensions, and layer normalization across channel+depth
    at each pooled spatial location.

    Computation steps in forward:
        1. Apply BatchNorm3d over (N, C, D, H, W).
        2. Merge channel and depth -> (N, C*D, H, W).
        3. Apply MaxPool2d over spatial dims (H, W).
        4. Reshape pooled result back to (N, H', W', C, D).
        5. Apply LayerNorm across the last two dimensions (C, D).
        6. Aggregate over depth (mean), producing (N, C, H', W').
    """
    def __init__(self, channels: int, depth: int, pool_kernel: int, pool_stride: int = None):
        """
        Initializes the layers.

        Args:
            channels (int): Number of channels C.
            depth (int): Number of depth slices D.
            pool_kernel (int): Kernel size for MaxPool2d.
            pool_stride (int, optional): Stride for MaxPool2d. If None, equals pool_kernel.
        """
        super(Model, self).__init__()
        self.channels = channels
        self.depth = depth
        self.pool_kernel = pool_kernel
        self.pool_stride = pool_stride if pool_stride is not None else pool_kernel

        # Batch normalization over 5D input (N, C, D, H, W)
        self.bn3d = nn.BatchNorm3d(num_features=self.channels)

        # MaxPool2d operates on 4D tensors (N, C', H, W) where we will set C' = C * D
        self.maxpool2d = nn.MaxPool2d(kernel_size=self.pool_kernel, stride=self.pool_stride)

        # LayerNorm will normalize over the (C, D) dimensions at each spatial location.
        # After pooling and reshaping we will have tensors shaped (..., C, D), so normalized_shape matches that.
        self.layernorm = nn.LayerNorm(normalized_shape=(self.channels, self.depth))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (N, C, H', W') where H', W' are pooled spatial sizes.
        """
        # 1) BatchNorm3d -> (N, C, D, H, W)
        x_bn = self.bn3d(x)

        N, C, D, H, W = x_bn.shape

        # 2) Merge channel and depth into channel dimension for 2D pooling:
        #    (N, C*D, H, W)
        merged = x_bn.reshape(N, C * D, H, W)

        # 3) Spatial max pooling on merged channels: (N, C*D, H', W')
        pooled = self.maxpool2d(merged)

        # 4) Compute pooled spatial sizes and reshape back to separate C and D:
        Hp, Wp = pooled.shape[2], pooled.shape[3]
        # Ensure the reshape matches the original C and D packing:
        reshaped = pooled.view(N, C, D, Hp, Wp).permute(0, 3, 4, 1, 2)  # (N, Hp, Wp, C, D)

        # 5) Apply LayerNorm across the last two dimensions (C, D)
        normalized = self.layernorm(reshaped)  # (N, Hp, Wp, C, D)

        # 6) Aggregate over the depth dimension (D) to reduce back to channels per spatial location
        aggregated = normalized.mean(dim=-1)  # (N, Hp, Wp, C)

        # Permute back to (N, C, Hp, Wp)
        out = aggregated.permute(0, 3, 1, 2).contiguous()

        return out

# Configuration variables
batch_size = 8
channels = 32
depth = 4
height = 64
width = 64
pool_kernel = 2
pool_stride = 2

def get_inputs():
    """
    Returns the runtime input tensors for the model.
    - x: tensor of shape (batch_size, channels, depth, height, width)
    """
    x = torch.randn(batch_size, channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters for the Model constructor:
    [channels, depth, pool_kernel, pool_stride]
    """
    return [channels, depth, pool_kernel, pool_stride]