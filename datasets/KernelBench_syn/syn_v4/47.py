import torch
import torch.nn as nn

"""
Complex 3D feature processing module combining InstanceNorm3d, a low-rank channel
projection (two linear layers applied per spatial location), ReLU6 nonlinearity,
AlphaDropout, and a residual connection scaled by an external channel-wise vector.

Structure follows the example style:
- Model class inheriting from nn.Module
- forward method implementing the computation
- get_inputs() returning runtime tensors for forward
- get_init_inputs() returning constructor parameters
- module-level configuration variables
"""

# Configuration / shapes
BATCH = 4
CHANNELS = 64
DEPTH = 8
HEIGHT = 16
WIDTH = 16
PROJ_RANK = 16  # rank for the low-rank channel projection

class Model(nn.Module):
    """
    Processes a 5D volumetric tensor (B, C, D, H, W) as follows:
      1. Instance normalization across channels per sample.
      2. For each spatial location, apply a low-rank channel projection:
         linear(C -> proj_rank) -> ReLU6 -> linear(proj_rank -> C)
      3. Multiply the reconstructed features by a channel-wise scale vector.
      4. Apply AlphaDropout for regularization.
      5. Add a residual connection to the original input.

    The low-rank projection is implemented via two nn.Linear layers applied to
    the channel dimension after flattening spatial locations so that the same
    projection is shared across all spatial positions and batch elements.
    """
    def __init__(self, in_channels: int, proj_rank: int, eps: float = 1e-5, dropout_p: float = 0.1):
        """
        Args:
            in_channels (int): Number of channels (C).
            proj_rank (int): Rank for the intermediate projection (smaller than C).
            eps (float): Epsilon for InstanceNorm3d.
            dropout_p (float): Dropout probability for AlphaDropout.
        """
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.proj_rank = proj_rank

        # Instance normalization across channels for 3D inputs
        self.instnorm = nn.InstanceNorm3d(num_features=in_channels, eps=eps, affine=True)

        # Low-rank projection: two linear layers acting on the channel dimension.
        # We'll apply these to shape (B * D * H * W, C).
        self.linear1 = nn.Linear(in_channels, proj_rank, bias=False)
        self.linear2 = nn.Linear(proj_rank, in_channels, bias=False)

        # Non-linearity and dropout
        self.relu6 = nn.ReLU6()
        self.alpha_dropout = nn.AlphaDropout(p=dropout_p)

        # Initialize projection matrices with a scaled orthogonal-like init for stability
        nn.init.kaiming_uniform_(self.linear1.weight, a=0, mode='fan_in', nonlinearity='linear')
        nn.init.kaiming_uniform_(self.linear2.weight, a=0, mode='fan_in', nonlinearity='linear')

    def forward(self, x: torch.Tensor, channel_scale: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, D, H, W).
            channel_scale (torch.Tensor): 1D tensor of shape (C,) to scale channels after reconstruction.

        Returns:
            torch.Tensor: Output tensor of shape (B, C, D, H, W).
        """
        B, C, D, H, W = x.shape
        assert C == self.in_channels, f"Expected input with {self.in_channels} channels, got {C}"
        assert channel_scale.shape == (C,), f"channel_scale must have shape ({C},), got {tuple(channel_scale.shape)}"

        # 1) Instance normalization
        y = self.instnorm(x)  # (B, C, D, H, W)

        # 2) Move channels to the last dimension and flatten spatial dims so we can apply linear layers
        y = y.permute(0, 2, 3, 4, 1).contiguous()  # (B, D, H, W, C)
        spatial_count = B * D * H * W
        y_flat = y.view(spatial_count, C)  # (B*D*H*W, C)

        # 3) Low-rank channel projection and reconstruction
        projected = self.linear1(y_flat)        # (B*D*H*W, proj_rank)
        activated = self.relu6(projected)       # (B*D*H*W, proj_rank)
        reconstructed = self.linear2(activated) # (B*D*H*W, C)

        # 4) Restore original shape (B, C, D, H, W)
        out = reconstructed.view(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()  # (B, C, D, H, W)

        # 5) Channel-wise scaling (broadcasted across spatial dims)
        scale = channel_scale.view(1, C, 1, 1, 1)
        out = out * scale

        # 6) AlphaDropout for regularization (keeps self-normalizing properties)
        out = self.alpha_dropout(out)

        # 7) Residual connection to the original input
        out = out + x

        return out

def get_inputs():
    """
    Returns runtime inputs for the forward method:
      - x: a random 5D tensor (BATCH, CHANNELS, DEPTH, HEIGHT, WIDTH)
      - channel_scale: a random 1D tensor of shape (CHANNELS,)
    """
    x = torch.randn(BATCH, CHANNELS, DEPTH, HEIGHT, WIDTH)
    # channel scale initialized close to 1.0 so residual behavior is preserved initially
    channel_scale = 1.0 + 0.05 * torch.randn(CHANNELS)
    return [x, channel_scale]

def get_init_inputs():
    """
    Returns initialization parameters for the Model constructor:
      - in_channels (int)
      - proj_rank (int)
    """
    return [CHANNELS, PROJ_RANK]