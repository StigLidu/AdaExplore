import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Any

# Module-level configuration
batch_size = 8
in_channels = 3
input_height = 64
input_width = 48

# Pooling output size (height, width)
pool_output_size: Tuple[int, int] = (16, 12)

# Patch / fold parameters
kernel_size: Tuple[int, int] = (3, 3)
stride: Tuple[int, int] = (2, 2)
padding: Tuple[int, int] = (1, 1)

# Threshold and gating scale
threshold_value: float = 0.05
gate_scale: float = 10.0

# Final projection dimensionality
out_features: int = 512


class Model(nn.Module):
    """
    Complex model that:
      - Applies AdaptiveAvgPool2d to reduce spatial resolution,
      - Thresholds small activations,
      - Extracts sliding local blocks (unfold), computes a per-patch gating,
      - Scales patches and reconstructs the pooled feature map with nn.Fold,
      - Flattens and projects to a final feature vector.

    This demonstrates combining AdaptiveAvgPool2d, Threshold, and Fold together
    with tensor manipulation to produce a non-trivial dataflow.
    """
    def __init__(
        self,
        in_channels: int,
        pool_output_size: Tuple[int, int],
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        padding: Tuple[int, int],
        threshold_value: float = 0.05,
        gate_scale: float = 10.0,
        out_features: int = 512,
    ):
        """
        Initializes pooling, thresholding, folding, and a final linear projection.

        Args:
            in_channels: Number of input channels.
            pool_output_size: Spatial size (H, W) after adaptive pooling.
            kernel_size: Kernel size for unfolding/folding (kH, kW).
            stride: Stride for unfolding/folding (sH, sW).
            padding: Padding for unfolding/folding (pH, pW).
            threshold_value: Value for nn.Threshold (elements below this become the 'value' supplied).
            gate_scale: Scalar to scale the per-patch mean before sigmoid gating.
            out_features: Dimensionality of the final projected output vector.
        """
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.pool_output_size = pool_output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.threshold_value = threshold_value
        self.gate_scale = gate_scale
        self.out_features = out_features

        # Layers
        self.pool = nn.AdaptiveAvgPool2d(self.pool_output_size)
        # Threshold(threshold, value) - values <= threshold become 'value'
        # We choose value=0.0 to zero out small activations
        self.threshold = nn.Threshold(self.threshold_value, 0.0)
        # Fold reconstructs the pooled spatial map from patches
        self.fold = nn.Fold(
            output_size=self.pool_output_size,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )

        # Linear projection parameters: project flattened folded output to out_features
        flattened_dim = self.in_channels * self.pool_output_size[0] * self.pool_output_size[1]
        self.proj_weight = nn.Parameter(torch.randn(flattened_dim, self.out_features) * (1.0 / (flattened_dim ** 0.5)))
        self.proj_bias = nn.Parameter(torch.zeros(self.out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Steps:
          1. Adaptive average pooling to (P, Q)
          2. Threshold small activations to 0
          3. Unfold pooled tensor into patches (N, C*kH*kW, L)
          4. Compute per-patch mean -> sigmoid(gate_scale * mean) (N, 1, L)
          5. Scale patches by gating and fold back to (N, C, P, Q)
          6. Flatten spatial and channels and apply a linear projection

        Args:
            x: Input tensor of shape (N, C, H, W)

        Returns:
            Tensor of shape (N, out_features)
        """
        # 1. Adaptive average pooling
        pooled = self.pool(x)  # shape: (N, C, P, Q)

        # 2. Threshold small activations
        # nn.Threshold operates elementwise; apply directly
        thr = self.threshold(pooled)  # shape: (N, C, P, Q)

        # 3. Unfold into patches
        # F.unfold expects kernel_size/stride/padding as ints or tuples
        patches = F.unfold(
            thr,
            kernel_size=self.kernel_size,
            dilation=1,
            padding=self.padding,
            stride=self.stride,
        )  # shape: (N, C * kH * kW, L)

        # 4. Per-patch gating
        # Compute mean across the patch vector dimension (C*kH*kW) -> (N, 1, L)
        patch_means = patches.mean(dim=1, keepdim=True)
        gating = torch.sigmoid(patch_means * self.gate_scale)  # (N, 1, L)

        # 5. Scale patches and fold back
        gated_patches = patches * gating  # broadcast over the channel*patch dimension
        folded = self.fold(gated_patches)  # shape: (N, C, P, Q)

        # 6. Flatten and linear projection
        N = folded.shape[0]
        flattened = folded.reshape(N, -1)  # shape: (N, C*P*Q)
        out = flattened.matmul(self.proj_weight) + self.proj_bias  # shape: (N, out_features)

        return out


def get_inputs() -> List[torch.Tensor]:
    """
    Prepares example input tensors to exercise the Model.

    Returns:
        A list containing a single tensor with shape (batch_size, in_channels, H, W).
    """
    x = torch.randn(batch_size, in_channels, input_height, input_width)
    return [x]


def get_init_inputs() -> List[Any]:
    """
    Returns initialization parameters for constructing the Model in the same order
    as Model.__init__ signature (excluding 'self').

    This allows external test harnesses to construct the Model with the same settings
    used to generate inputs.
    """
    return [
        in_channels,               # in_channels
        pool_output_size,          # pool_output_size
        kernel_size,               # kernel_size
        stride,                    # stride
        padding,                   # padding
        threshold_value,           # threshold_value
        gate_scale,                # gate_scale
        out_features,              # out_features
    ]