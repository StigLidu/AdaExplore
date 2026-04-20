import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex 3D feature processing module that combines lazy batch normalization,
    3D average pooling, sigmoid gating and a learned channel mixing via matrix multiplication.

    Computation steps:
    1. Apply LazyBatchNorm3d to input (lazy init on first forward pass).
    2. Apply 3D average pooling to reduce spatial dimensions.
    3. Compute a channel-wise gate by global spatial average followed by a Sigmoid.
    4. Apply the gate to pooled features (channel-wise modulation).
    5. Aggregate spatial locations into channel descriptors and apply a channel mixing
       using an external projection matrix (provided as an input).
    6. L2-normalize the resulting channel vectors per sample.

    The model accepts an initialization parameter 'pool_kernel' which determines the
    AvgPool3d kernel/stride and 'eps' forwarded to LazyBatchNorm3d.
    """
    def __init__(self, pool_kernel: int = 2, eps: float = 1e-5):
        """
        Args:
            pool_kernel (int): Kernel size and stride for AvgPool3d.
            eps (float): Epsilon for LazyBatchNorm3d numerical stability.
        """
        super(Model, self).__init__()
        # 3D average pooling (reduces D, H, W by pool_kernel)
        self.pool = nn.AvgPool3d(kernel_size=pool_kernel, stride=pool_kernel)
        # LazyBatchNorm3d will infer num_features from the first input during forward
        self.bn = nn.LazyBatchNorm3d(eps=eps)
        # Sigmoid gating for channel-wise modulation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, channel_proj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, D, H, W).
            channel_proj (torch.Tensor): Channel mixing matrix of shape (C, C_out)
                                         (C_out can be equal to C for square mixing).

        Returns:
            torch.Tensor: Output tensor of shape (B, C_out) representing
                          per-sample channel descriptors after mixing and normalization.
        """
        # Apply lazy batch normalization (will initialize running stats/affine lazily)
        x_bn = self.bn(x)  # (B, C, D, H, W)

        # Average pool to reduce spatial resolution
        x_pooled = self.pool(x_bn)  # (B, C, D', H', W')

        # Global spatial average to compute channel-wise gating
        # Keep dims for broadcasting back into pooled tensor
        gate = x_pooled.mean(dim=(2, 3, 4), keepdim=True)  # (B, C, 1, 1, 1)
        gate = self.sigmoid(gate)  # (B, C, 1, 1, 1)

        # Modulate pooled features by the gate
        x_gated = x_pooled * gate  # broadcasting over spatial dims -> (B, C, D', H', W')

        # Collapse spatial dims into one and aggregate per-channel descriptors
        B, C, Dp, Hp, Wp = x_gated.shape
        x_flat = x_gated.view(B, C, -1)  # (B, C, L) where L = D'*H'*W'
        channel_desc = x_flat.sum(dim=2)  # (B, C) -- aggregated channel descriptors

        # Mix channels using the provided projection/mixing matrix
        # channel_proj shape: (C, C_out) so result is (B, C_out)
        mixed = torch.matmul(channel_desc, channel_proj)

        # L2-normalize per-sample to keep scale stable
        norm = mixed.norm(p=2, dim=1, keepdim=True)  # (B, 1)
        output = mixed / (norm + 1e-6)

        return output

# Configuration variables
batch_size = 8
channels = 32
depth = 16   # D
height = 32  # H
width = 32   # W
pool_kernel = 2
eps = 1e-5

def get_inputs():
    """
    Returns example inputs for the model:
      - x: random 5D tensor (B, C, D, H, W)
      - channel_proj: random channel mixing matrix (C, C) for square mixing
    """
    x = torch.randn(batch_size, channels, depth, height, width)
    # channel mixing matrix: square (channels x channels) in this example
    channel_proj = torch.randn(channels, channels)
    return [x, channel_proj]

def get_init_inputs():
    """
    Returns initialization parameters for the Model constructor in order:
      [pool_kernel, eps]
    """
    return [pool_kernel, eps]