import torch
import torch.nn as nn

# Configuration variables
batch_size = 4
in_channels = 3
depth = 16
height = 32
width = 32
final_features = 32  # dimensionality of final linear projection

class Model(nn.Module):
    """
    A more complex 3D-processing module that:
    - Uses two parallel LazyConv3d branches (different receptive fields)
    - Applies SiLU activations per branch
    - Concatenates branch outputs along the channel dimension
    - Applies a 3D average pooling to reduce spatial resolution
    - Projects concatenated channels down with a 1x1 Conv3d
    - Applies SiLU, performs global spatial averaging, and applies a final linear projection
    This design showcases LazyConv3d, AvgPool3d, and SiLU in a small residual/aggregation pattern.
    """
    def __init__(self):
        super(Model, self).__init__()
        # Two lazy 3D conv branches with different kernels -> in_channels will be inferred at first forward
        self.branch_a = nn.LazyConv3d(out_channels=8, kernel_size=3, padding=1, stride=1)
        self.branch_b = nn.LazyConv3d(out_channels=12, kernel_size=1, padding=0, stride=1)

        # Activation
        self.act = nn.SiLU()

        # 3D average pooling to downsample spatial dims by factor 2
        self.pool = nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # After concatenation (8 + 12 = 20 channels) we reduce to 16 channels with a pointwise conv
        # in_channels is known (20) so a regular Conv3d is fine here
        self.reduce_conv = nn.Conv3d(in_channels=20, out_channels=16, kernel_size=1, stride=1)

        # Final linear projection parameters (applied after global spatial average)
        # We use explicit Parameters to avoid assuming a specific nn.Linear shape initialization method
        self.fc_weight = nn.Parameter(torch.randn(final_features, 16))  # (out_features, in_features)
        self.fc_bias = nn.Parameter(torch.randn(final_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, C, D, H, W)

        Returns:
            Tensor of shape (B, final_features)
        """
        # Branch convolutions
        a = self.branch_a(x)    # -> (B, 8, D, H, W)
        b = self.branch_b(x)    # -> (B, 12, D, H, W)

        # Non-linearities
        a = self.act(a)
        b = self.act(b)

        # Concatenate along channel dimension
        concat = torch.cat([a, b], dim=1)  # -> (B, 20, D, H, W)

        # Downsample spatial dimensions
        pooled = self.pool(concat)  # -> (B, 20, D/2, H/2, W/2)

        # Channel reduction
        reduced = self.reduce_conv(pooled)  # -> (B, 16, D/2, H/2, W/2)
        reduced = self.act(reduced)

        # Global spatial average to get a fixed-size feature vector per batch element
        # Average over D, H, W
        feat = reduced.mean(dim=(2, 3, 4))  # -> (B, 16)

        # Final linear projection via explicit matmul with parameters
        out = torch.matmul(feat, self.fc_weight.t()) + self.fc_bias  # -> (B, final_features)

        return out

def get_inputs():
    # Create a random 5D tensor representing a batch of volumetric data (e.g., small video clips)
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    # No special initialization inputs required; LazyConv3d will infer input channels on first forward
    return []