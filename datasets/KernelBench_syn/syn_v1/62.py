import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    A moderately complex 3D feature processing module that:
      - Applies 3D average pooling to reduce spatial resolution.
      - Applies a Hardshrink nonlinearity to encourage sparsity.
      - Projects per-voxel channel features through a small MLP (implemented as two linear layers).
      - Uses AlphaDropout for regularization on the hidden features.
      - Reconstructs a channel-wise feature map, scales it with a learnable per-channel parameter,
        upsamples it back to the original spatial size, and adds a residual 1x1x1 projection from the input.
    The combination demonstrates spatial pooling, sparse activation, channel projection, dropout, and upsampling.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride,
        padding,
        hardshrink_lambda: float,
        dropout_p: float,
        hidden_dim: int
    ):
        """
        Initializes the Model.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels after reconstruction.
            kernel_size (int or tuple): Kernel size for AvgPool3d.
            stride (int or tuple): Stride for AvgPool3d.
            padding (int or tuple): Padding for AvgPool3d.
            hardshrink_lambda (float): Lambda parameter for Hardshrink activation.
            dropout_p (float): Dropout probability for AlphaDropout.
            hidden_dim (int): Hidden dimension of the per-voxel MLP.
        """
        super(Model, self).__init__()
        # 3D average pooling to reduce spatial resolution
        self.avgpool = nn.AvgPool3d(kernel_size=kernel_size, stride=stride, padding=padding)
        # Sparse activation
        self.hardshrink = nn.Hardshrink(lambd=hardshrink_lambda)
        # Per-voxel MLP: channel -> hidden -> out_channels
        # We'll apply these via Linear on the last dimension after reshaping
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.dropout = nn.AlphaDropout(p=dropout_p)
        self.fc2 = nn.Linear(hidden_dim, out_channels)
        # Learnable per-channel scale applied after reconstruction (broadcastable)
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))
        # Residual 1x1x1 convolution to match input channels to out_channels for skip connection
        self.res_proj = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        # Small epsilon for numerical stability in a normalization step used optionally
        self.eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C_in, D, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (N, C_out, D, H, W)
        """
        # Save original spatial sizes for final upsampling
        N, C_in, D, H, W = x.shape

        # 1) Spatial reduction
        pooled = self.avgpool(x)  # (N, C_in, Dp, Hp, Wp)

        # 2) Sparse activation to zero-out small values
        activated = self.hardshrink(pooled)  # (N, C_in, Dp, Hp, Wp)

        # 3) Prepare per-voxel features: move channels to last dim and flatten spatial
        #    (N, Dp, Hp, Wp, C_in) -> (N, voxels, C_in)
        perm = activated.permute(0, 2, 3, 4, 1).contiguous()
        Np, Dp, Hp, Wp, Cin = perm.shape
        voxels = Dp * Hp * Wp
        flat = perm.view(Np, voxels, Cin)  # (N, voxels, C_in)

        # 4) Per-voxel MLP: Linear -> AlphaDropout -> ReLU -> Linear
        hidden = self.fc1(flat)           # (N, voxels, hidden_dim)
        hidden = self.dropout(hidden)     # AlphaDropout
        hidden = torch.relu(hidden)
        out_feats = self.fc2(hidden)      # (N, voxels, out_channels)

        # 5) Reshape back to (N, out_channels, Dp, Hp, Wp)
        out_feats = out_feats.view(Np, Dp, Hp, Wp, -1).permute(0, 4, 1, 2, 3).contiguous()

        # 6) Channel-wise scaling
        out_scaled = out_feats * self.scale  # broadcast over spatial dims

        # 7) Upsample reconstructed feature map back to original spatial dimensions
        #    Use trilinear interpolation for smoothness
        up = F.interpolate(out_scaled, size=(D, H, W), mode='trilinear', align_corners=False)

        # 8) Residual projection of the input to match out_channels and add
        res = self.res_proj(x)  # (N, out_channels, D, H, W)
        combined = up + res

        # 9) Small normalization: channel-wise L2 normalization to avoid exploding magnitudes,
        #    then a bounded non-linearity to keep values stable.
        #    Compute L2 norm over spatial dims per channel and normalize.
        #    norm shape: (N, C_out, 1, 1, 1)
        norm = torch.sqrt(torch.sum(combined * combined, dim=(2, 3, 4), keepdim=True) + self.eps)
        normalized = combined / norm

        # Final output bounded with tanh to keep the dynamic range limited
        return torch.tanh(normalized)

# Module-level configuration variables
batch_size = 4
in_channels = 16
out_channels = 24
depth = 16
height = 32
width = 32

# Pooling configuration (3D)
kernel_size = (2, 2, 2)
stride = (2, 2, 2)
padding = (0, 0, 0)

# Nonlinearity and dropout params
hardshrink_lambda = 0.5
dropout_p = 0.1
hidden_dim = 64

def get_inputs():
    """
    Returns a list with a single input tensor of shape (batch_size, in_channels, depth, height, width).
    """
    x = torch.randn(batch_size, in_channels, depth, height, width, dtype=torch.float32)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters for the Model in the same order as the __init__ signature:
      in_channels, out_channels, kernel_size, stride, padding, hardshrink_lambda, dropout_p, hidden_dim
    """
    return [in_channels, out_channels, kernel_size, stride, padding, hardshrink_lambda, dropout_p, hidden_dim]