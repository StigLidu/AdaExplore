import torch
import torch.nn as nn

# Configuration / module-level variables
BATCH_SIZE = 8
CHANNELS = 16
DEPTH = 8
HEIGHT = 32
WIDTH = 32

POOL_KERNEL = (2, 2, 2)        # Kernel/stride for AvgPool3d
LINEAR_OUT_FEATURES = 1024     # Output features for LazyLinear
SOFTSHRINK_LAMBDA = 0.5        # Lambda parameter for Softshrink

class Model(nn.Module):
    """
    Complex example combining AvgPool3d, Softshrink, and LazyLinear.

    Pipeline:
    - Input x: (batch, channels, D, H, W)
    - 3D average pooling to reduce spatial dims
    - Softshrink activation on pooled tensor
    - Flatten spatial dims into feature vector per batch
    - LazyLinear projection to a fixed output dimension (in_features inferred on first forward)
    - Gating using a per-sample scalar condition `cond` (broadcasted)
    - Another Softshrink and a broadcasted residual from the flattened features' global mean
    - Output: (batch, out_features)
    """
    def __init__(
        self,
        out_features: int = LINEAR_OUT_FEATURES,
        softshrink_lambda: float = SOFTSHRINK_LAMBDA,
        pool_kernel = POOL_KERNEL
    ):
        super(Model, self).__init__()
        # 3D average pooling reduces (D,H,W) by pool_kernel (kernel_size=stride)
        self.pool = nn.AvgPool3d(kernel_size=pool_kernel, stride=pool_kernel)
        # Element-wise soft shrink activation
        self.softshrink = nn.Softshrink(lambd=softshrink_lambda)
        # LazyLinear will infer in_features at first forward pass
        self.proj = nn.LazyLinear(out_features)
        self.out_features = out_features

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, channels, D, H, W)
            cond: Tensor of shape (batch, 1) used as a gating scalar per sample

        Returns:
            Tensor of shape (batch, out_features)
        """
        # 1) Reduce spatial dimensions via average pooling
        pooled = self.pool(x)  # (batch, channels, D', H', W')

        # 2) Apply non-linear soft shrink activation
        activated = self.softshrink(pooled)  # same shape as pooled

        # 3) Flatten channels and spatial dims into a single feature vector
        flattened = activated.flatten(start_dim=1)  # (batch, F)

        # 4) Project to target feature dimension with a LazyLinear (in_features inferred)
        projected = self.proj(flattened)  # (batch, out_features)

        # 5) Gating: use cond (batch,1) to modulate projected features (broadcast)
        gate = torch.sigmoid(cond)  # keep values in (0,1)
        gated = projected * gate  # (batch, out_features) broadcast multiplication

        # 6) Another Softshrink to promote sparsity / shrink small activations
        shrunk = self.softshrink(gated)  # (batch, out_features)

        # 7) Broadcast a global residual computed from the flattened features' mean
        #    This introduces a lightweight global context per sample.
        global_context = flattened.mean(dim=1, keepdim=True)  # (batch, 1)
        output = shrunk + global_context  # broadcast to (batch, out_features)

        return output


def get_inputs():
    """
    Returns:
        [x, cond] where:
        - x: Tensor of shape (BATCH_SIZE, CHANNELS, DEPTH, HEIGHT, WIDTH)
        - cond: Tensor of shape (BATCH_SIZE, 1)
    """
    x = torch.randn(BATCH_SIZE, CHANNELS, DEPTH, HEIGHT, WIDTH, dtype=torch.float32)
    # Use a small-range conditioning scalar per sample (can be negative/positive)
    cond = torch.randn(BATCH_SIZE, 1, dtype=torch.float32)
    return [x, cond]


def get_init_inputs():
    """
    Returns initialization parameters that can be used to construct the Model.
    """
    return [LINEAR_OUT_FEATURES, SOFTSHRINK_LAMBDA, POOL_KERNEL]