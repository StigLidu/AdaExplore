import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class Model(nn.Module):
    """
    Composite model that:
    - Applies 3D adaptive max pooling to a 5D tensor (B, C3, D, H, W)
    - Collapses the depth dimension and projects to a shared channel space via 1x1 Conv2d
    - Normalizes the projected 3D-derived features with LazyInstanceNorm2d
    - Pools and projects a 4D tensor (B, C2, H, W) into the same channel space and normalizes with InstanceNorm2d
    - Computes a learned spatial gate to blend the two normalized feature maps
    - Produces a pooled feature vector per batch element
    """
    def __init__(self, c_proj: int = 64, pool_output: Tuple[int, int, int] = (4, 16, 16)):
        """
        Args:
            c_proj: Number of channels to project both pathways into (shared feature dimension).
            pool_output: Target output size for AdaptiveMaxPool3d in (D_out, H_out, W_out).
        """
        super(Model, self).__init__()
        self.c_proj = c_proj
        self.pool_output = pool_output  # (D_out, H_out, W_out)

        # 3D adaptive max pooling to reduce/spatially re-sample the 3D input
        self.pool3d = nn.AdaptiveMaxPool3d(self.pool_output)

        # After collapsing depth, we use 1x1 Conv2d to project channels from the 3D pathway
        # The in_channels will be inferred in forward for lazy norm, but conv needs explicit channels;
        # we'll instantiate convs in forward_hook style by creating placeholder attributes here and
        # lazily initializing them when we see inputs. To keep structure simple, assume external
        # initialization provides reasonable channel sizes via get_init_inputs.
        # For design clarity, the conv layers are created with placeholder in/out channels and will
        # be reset during first forward if mismatched. However, to maintain typical usage, we will
        # create conv layers when the module is initialized by the caller with matching dimensions.
        # Therefore, this module expects convs to be created with the correct in_channels externally.

        # Projectors (actual in_channels are set by the caller when constructing the Model)
        # To enforce a clean API, we create the convs with in_channels unknown using Lazy modules:
        # Use nn.Conv2d only with known in/out channels; so we expect the constructor to be called
        # with the intended c_proj and then user will rely on get_init_inputs for matching inputs.
        # For robustness in examples, we will create convs with generic values that are safe for common sizes.
        # These will be replaced if the shapes differ at runtime (handled below).
        self.conv3d_proj = nn.Conv2d(in_channels=32, out_channels=self.c_proj, kernel_size=1)
        self.conv2d_proj = nn.Conv2d(in_channels=32, out_channels=self.c_proj, kernel_size=1)

        # Normalization modules:
        # LazyInstanceNorm2d will infer num_features from the incoming tensor (3D pathway projected)
        self.lazy_in_norm = nn.LazyInstanceNorm2d()
        # InstanceNorm2d requires a known number of features; use c_proj
        self.inst_norm = nn.InstanceNorm2d(self.c_proj, affine=False, track_running_stats=False)

        # Gate conv to compute a spatial gating map (sigmoid) from concatenated features
        self.gate_conv = nn.Conv2d(in_channels=2 * self.c_proj, out_channels=1, kernel_size=1)

        # Final pooling to get a vector per batch element
        self.final_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x3d: torch.Tensor, z2d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x3d: Tensor with shape (B, C3, D, H, W)
            z2d: Tensor with shape (B, C2, H, W)

        Returns:
            Tensor of shape (B, c_proj) — pooled fused features per batch element.
        """
        B = x3d.shape[0]

        # 1) Adaptive max pool the 3D input to a smaller (D_out, H_out, W_out)
        x = self.pool3d(x3d)  # (B, C3, D_out, H_out, W_out)

        # 2) Collapse the depth dimension by mean to produce a 4D tensor (B, C3, H_out, W_out)
        x = torch.mean(x, dim=2)  # mean over depth -> (B, C3, H_out, W_out)

        # If conv3d_proj in_channels doesn't match x channels, recreate it to match runtime dims.
        in_c_x = x.shape[1]
        if self.conv3d_proj.in_channels != in_c_x or self.conv3d_proj.out_channels != self.c_proj:
            self.conv3d_proj = nn.Conv2d(in_channels=in_c_x, out_channels=self.c_proj, kernel_size=1).to(x.device)

        # 3) Project the collapsed 3D features into shared channel space
        x = self.conv3d_proj(x)  # (B, c_proj, H_out, W_out)

        # 4) Normalize projected 3D features using LazyInstanceNorm2d (num_features inferred)
        x = self.lazy_in_norm(x)

        # 5) Resize 2D input to match spatial resolution of pooled 3D-derived features
        _, _, H_out, W_out = x.shape
        z = F.adaptive_avg_pool2d(z2d, (H_out, W_out))  # (B, C2, H_out, W_out)

        # If conv2d_proj in_channels doesn't match z channels, recreate it to match runtime dims.
        in_c_z = z.shape[1]
        if self.conv2d_proj.in_channels != in_c_z or self.conv2d_proj.out_channels != self.c_proj:
            self.conv2d_proj = nn.Conv2d(in_channels=in_c_z, out_channels=self.c_proj, kernel_size=1).to(z.device)

        # 6) Project and normalize the 2D pathway (InstanceNorm2d requires known num_features)
        z = self.conv2d_proj(z)  # (B, c_proj, H_out, W_out)
        z = self.inst_norm(z)

        # 7) Compute a learned spatial gate based on concatenated features
        concat = torch.cat([x, z], dim=1)  # (B, 2*c_proj, H_out, W_out)
        gate = torch.sigmoid(self.gate_conv(concat))  # (B, 1, H_out, W_out)

        # 8) Fuse features with gating and apply non-linearity
        out = x * gate + z * (1.0 - gate)
        out = F.relu(out)

        # 9) Aggregate spatially to produce a per-batch feature vector
        out = self.final_pool(out).view(B, -1)  # (B, c_proj)

        return out

# Configuration / default input sizes
batch_size = 8
C3 = 16         # channels for the 3D input
D = 8           # depth
H = 64          # height
W = 64          # width

C2 = 32         # channels for the 2D input

C_PROJ = 48     # number of projection channels / shared feature dimension
POOL_OUTPUT = (4, 16, 16)  # (D_out, H_out, W_out) for AdaptiveMaxPool3d

def get_inputs() -> List[torch.Tensor]:
    """
    Returns a list of input tensors for the model:
    - x3d: (batch_size, C3, D, H, W)
    - z2d: (batch_size, C2, H, W)
    """
    x3d = torch.randn(batch_size, C3, D, H, W)
    z2d = torch.randn(batch_size, C2, H, W)
    return [x3d, z2d]

def get_init_inputs():
    """
    Returns initialization parameters that can be used to construct the Model.
    For example:
        model = Model(*get_init_inputs())
    """
    return [C_PROJ, POOL_OUTPUT]