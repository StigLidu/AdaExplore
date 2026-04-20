import torch
import torch.nn as nn

# Configuration
batch_size = 32
channels = 64            # Must be divisible by 2 for channel-splitting
height = 16
width = 128

pool_kernel = 3
pool_stride = 2

# Derived sizes (used to initialize bilinear)
c1 = channels // 2
c2 = channels - c1
pooled_length = (width - pool_kernel) // pool_stride + 1  # length after AvgPool1d
in1_features = c1 * pooled_length
in2_features = c2 * pooled_length
bilinear_out_features = 512

class Model(nn.Module):
    """
    Complex module that:
    - Applies a LazyInstanceNorm2d over a 4D input (B, C, H, W)
    - Collapses the height dimension via mean to produce (B, C, W)
    - Uses a 1D average pooling to reduce the width dimension
    - Splits channels into two groups, flattens each, and feeds them to a Bilinear layer
    - Applies a final non-linear activation

    Overall computation pattern intentionally mixes normalization, pooling across spatial
    dimensions, channel splitting, and a bilinear interaction between two pooled views.
    """
    def __init__(self):
        super(Model, self).__init__()
        # Lazy instance norm will learn per-channel affine parameters after first forward
        self.instnorm = nn.LazyInstanceNorm2d()
        # 1D average pooling applied along the width dimension (after collapsing height)
        self.avgpool1d = nn.AvgPool1d(kernel_size=pool_kernel, stride=pool_stride)
        # Bilinear layer combines two flattened pooled channel groups
        self.bilinear = nn.Bilinear(in1_features, in2_features, bilinear_out_features, bias=True)
        # Small final projection to keep outputs compact and stable
        self.final_proj = nn.Linear(bilinear_out_features, bilinear_out_features)
        self.activation = torch.tanh

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, channels, height, width)

        Returns:
            Tensor of shape (batch_size, bilinear_out_features)
        """
        # 1) Normalize per-instance per-channel
        x_norm = self.instnorm(x)  # (B, C, H, W)

        # 2) Collapse height dimension by averaging to emphasize width-wise features
        # Result shape: (B, C, W)
        x_collapsed = x_norm.mean(dim=2)

        # 3) Apply 1D average pooling along the width dimension
        # AvgPool1d expects (B, C, L) and returns (B, C, L')
        x_pooled = self.avgpool1d(x_collapsed)  # (B, C, pooled_length)

        # 4) Split channels into two groups for a bilinear interaction
        x_a = x_pooled[:, :c1, :]  # (B, c1, pooled_length)
        x_b = x_pooled[:, c1:, :]  # (B, c2, pooled_length)

        # 5) Flatten each group's spatial dimensions into feature vectors
        x_a_flat = x_a.reshape(x_a.size(0), -1)  # (B, in1_features)
        x_b_flat = x_b.reshape(x_b.size(0), -1)  # (B, in2_features)

        # 6) Bilinear interaction between the two flattened representations
        bilinear_out = self.bilinear(x_a_flat, x_b_flat)  # (B, bilinear_out_features)

        # 7) Final projection and non-linear activation
        out = self.final_proj(bilinear_out)
        out = self.activation(out)
        return out

def get_inputs():
    # Random input following the configured shapes
    x = torch.randn(batch_size, channels, height, width, dtype=torch.float32)
    return [x]

def get_init_inputs():
    # Model has no required constructor arguments (LazyInstanceNorm2d is lazy),
    # so no initialization inputs are needed.
    return []