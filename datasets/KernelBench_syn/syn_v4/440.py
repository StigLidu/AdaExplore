import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex 3D spatial feature processor combining InstanceNorm3d, LocalResponseNorm, and
    two Linear projections applied across the channel dimension for each spatial location.

    Computation pipeline:
      1. InstanceNorm3d over (N, C, D, H, W)
      2. LocalResponseNorm across channels
      3. Move channels to the last dimension and flatten spatial dims -> (N, S, C)
      4. Linear expansion: (N, S, C) -> (N, S, C * out_multiplier)
      5. GELU activation
      6. Linear projection back to original channels: (N, S, C * out_multiplier) -> (N, S, C)
      7. Reshape back to (N, C, D, H, W) and add a residual connection to the normalized input
    """
    def __init__(self, in_channels: int, out_multiplier: int = 2, inst_eps: float = 1e-5, inst_affine: bool = True):
        """
        Args:
            in_channels (int): Number of input channels (C).
            out_multiplier (int): Expansion multiplier for the intermediate linear feature size.
            inst_eps (float): Epsilon for InstanceNorm3d.
            inst_affine (bool): Whether InstanceNorm3d has affine parameters.
        """
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.out_multiplier = out_multiplier
        mid_channels = in_channels * out_multiplier

        # Instance normalization over 3D volumes
        self.inst_norm = nn.InstanceNorm3d(num_features=in_channels, eps=inst_eps, affine=inst_affine)

        # Local response normalization across channels (works for N, C, D, H, W)
        self.lrn = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0)

        # Two linear layers applied to the channel dimension at each spatial location:
        # We will reshape (N, C, D, H, W) -> (N, S, C) where S = D*H*W, and apply these linears
        self.expand_fc = nn.Linear(in_channels, mid_channels, bias=True)
        self.project_fc = nn.Linear(mid_channels, in_channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor with shape (N, C, D, H, W).

        Returns:
            torch.Tensor: Output tensor with same shape (N, C, D, H, W).
        """
        N, C, D, H, W = x.shape
        # 1) Instance normalization
        inst = self.inst_norm(x)  # (N, C, D, H, W)

        # 2) Local response normalization across channels
        lrn_out = self.lrn(inst)  # (N, C, D, H, W)

        # 3) Move channels to last dim and flatten spatial dims -> (N, S, C)
        # Permute to (N, D, H, W, C)
        perm = lrn_out.permute(0, 2, 3, 4, 1).contiguous()
        S = D * H * W
        seq = perm.view(N, S, C)  # (N, S, C)

        # 4) Linear expansion across channels
        expanded = self.expand_fc(seq)  # (N, S, C * out_multiplier)

        # 5) Non-linearity
        activated = F.gelu(expanded)

        # 6) Project back to original channel dimension
        projected = self.project_fc(activated)  # (N, S, C)

        # 7) Reshape back to (N, C, D, H, W)
        reshaped = projected.view(N, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()  # (N, C, D, H, W)

        # Residual connection with the instance-normalized tensor
        out = reshaped + inst

        return out

# Configuration variables
batch_size = 8
channels = 32
depth = 8
height = 16
width = 16
out_multiplier = 2

def get_inputs():
    """
    Returns:
        List containing a single 5D input tensor of shape (batch_size, channels, depth, height, width).
    """
    x = torch.randn(batch_size, channels, depth, height, width, dtype=torch.float32)
    return [x]

def get_init_inputs():
    """
    Returns:
        List of initialization inputs for the Model constructor: [in_channels, out_multiplier]
    """
    return [channels, out_multiplier]