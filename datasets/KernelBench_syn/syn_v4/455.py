import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A more complex 3D processing module that demonstrates padding, group normalization,
    hardsigmoid activation, global spatial pooling, and a projected gating mechanism.

    Computation steps:
      1. Zero-pad the 3D spatial dimensions.
      2. Apply GroupNorm across channels.
      3. Apply Hardsigmoid non-linearity element-wise.
      4. Global average pool over (D, H, W) to produce (N, C).
      5. Project pooled features into a lower-dimensional space via nn.Linear.
      6. Compute a gate using Hardsigmoid on the projected features and apply it multiplicatively.
      7. Return the gated projection.
    """
    def __init__(self, num_groups: int, proj_dim: int):
        """
        Args:
            num_groups (int): Number of groups for GroupNorm. Must divide global 'channels'.
            proj_dim (int): Output dimensionality of the final projection.
        """
        super(Model, self).__init__()
        # Use global channels defined at module level
        self.num_groups = num_groups
        self.proj_dim = proj_dim

        # Zero pad 3D spatial dims: (padL, padR, padT, padB, padF, padBack)
        # This will add 1 voxel padding on left/right (W), 2 on top/bottom (H), and 1 on front/back (D).
        self.pad = nn.ZeroPad3d((1, 1, 2, 2, 1, 1))

        # GroupNorm across channels
        self.gn = nn.GroupNorm(self.num_groups, channels)

        # Hardsigmoid activation (used twice: pre-pool and as gating nonlinearity)
        self.act = nn.Hardsigmoid()

        # Projection from pooled channels to projection dimension
        self.proj = nn.Linear(channels, self.proj_dim, bias=True)

        # Small learnable bias for gating scaling to increase expressive power
        self.gate_scale = nn.Parameter(torch.ones(self.proj_dim) * 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (N, proj_dim) (gated projection).
        """
        # 1) Zero-pad spatial dimensions
        x = self.pad(x)

        # 2) Group normalization across channels
        x = self.gn(x)

        # 3) Element-wise Hardsigmoid
        x = self.act(x)

        # 4) Global average pooling over D, H, W -> shape (N, C)
        x_pooled = x.mean(dim=(2, 3, 4))

        # 5) Linear projection to lower-dimensional representation -> shape (N, proj_dim)
        proj_out = self.proj(x_pooled)

        # 6) Compute gating values and apply elementwise gating
        gate = self.act(proj_out * self.gate_scale)  # scaled hardsigmoid gate
        gated = proj_out * gate

        return gated


# Module-level configuration variables
batch_size = 4
channels = 48     # must be divisible by num_groups
D = 8
H = 16
W = 16
num_groups = 8    # divides channels (48 % 8 == 0)
proj_dim = 128


def get_inputs():
    """
    Returns the input tensors for the model's forward pass.
    """
    x = torch.randn(batch_size, channels, D, H, W)
    return [x]


def get_init_inputs():
    """
    Returns the initialization parameters for constructing the Model.
    """
    return [num_groups, proj_dim]