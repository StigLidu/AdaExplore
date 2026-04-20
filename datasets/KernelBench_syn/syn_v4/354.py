import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex model combining spatial centering, 2D max pooling, 1D lazy batch normalization,
    ReLU activation, and channel-wise normalization to produce a compact per-channel descriptor.

    Computation steps:
    1. Center each channel by subtracting its spatial mean.
    2. Apply 2D max pooling to reduce spatial dimensions.
    3. Flatten the spatial dimensions and apply LazyBatchNorm1d across channels.
    4. Apply ReLU non-linearity.
    5. Aggregate over the spatial dimension to get per-channel descriptors.
    6. L2-normalize the per-channel descriptors per sample.
    """
    def __init__(self, pool_kernel: int = 2, eps: float = 1e-6):
        """
        Args:
            pool_kernel (int): Kernel (and stride) size for MaxPool2d.
            eps (float): Small epsilon for numerical stability when normalizing.
        """
        super(Model, self).__init__()
        self.pool_kernel = pool_kernel
        self.eps = eps

        # MaxPool2d for spatial downsampling
        self.pool = nn.MaxPool2d(kernel_size=self.pool_kernel, stride=self.pool_kernel)

        # LazyBatchNorm1d will infer num_features (channels) on first forward.
        # It expects input of shape (N, C) or (N, C, L); we'll provide (N, C, L).
        self.bn = nn.LazyBatchNorm1d()

        # Simple non-linearity
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Per-sample, per-channel L2-normalized descriptor of shape (N, C).
        """
        # 1) Spatial centering: subtract mean per-channel per-sample
        # mean_spatial has shape (N, C, 1, 1)
        mean_spatial = x.mean(dim=(2, 3), keepdim=True)
        x_centered = x - mean_spatial

        # 2) Spatial downsampling via MaxPool2d -> shape (N, C, H', W')
        x_pooled = self.pool(x_centered)

        # 3) Flatten spatial dims to prepare for BatchNorm1d: (N, C, L)
        N, C, Hp, Wp = x_pooled.shape
        x_flat = x_pooled.view(N, C, Hp * Wp)

        # 4) Apply LazyBatchNorm1d which normalizes across the channel dimension
        x_bn = self.bn(x_flat)

        # 5) Non-linearity
        x_act = self.relu(x_bn)

        # 6) Aggregate over the spatial dimension to produce per-channel descriptors: (N, C)
        channel_descr = x_act.mean(dim=2)

        # 7) Per-sample L2 normalization across channels
        norms = channel_descr.norm(p=2, dim=1, keepdim=True)  # (N, 1)
        channel_descr_normed = channel_descr / (norms + self.eps)

        return channel_descr_normed

# Configuration (module-level)
batch_size = 8
channels = 16
height = 64
width = 48
pool_kernel = 2
eps = 1e-6

def get_inputs():
    """
    Returns:
        list: [x] where x is a random tensor of shape (batch_size, channels, height, width)
    """
    x = torch.randn(batch_size, channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns:
        list: Initialization parameters for Model: [pool_kernel, eps]
    """
    return [pool_kernel, eps]