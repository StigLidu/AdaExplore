import torch
import torch.nn as nn
import torch.nn.functional as F

# Configuration / default sizes
batch_size = 8
in_channels = 16
mid_channels = 64
out_channels = 32
depth = 16
height = 32
width = 32

# Normalization / dropout / LRN defaults
num_groups = 8           # Must divide mid_channels
dropout_p = 0.2
lrn_size = 5
lrn_alpha = 1e-4
lrn_beta = 0.75
lrn_k = 1.0

# Final classifier size
fc_out_features = 128


class Model(nn.Module):
    """
    A moderately complex 3D feature extractor that combines Conv3d, GroupNorm,
    Dropout3d, and LocalResponseNorm in a residual-like block followed by
    global pooling and a fully-connected projection.

    Forward computation pattern:
        1) conv1 (3x3x3) -> GroupNorm -> ReLU
        2) Dropout3d
        3) conv2 (1x1x1) -> ReLU
        4) skip connection via 1x1x1 conv from input to match channels
        5) elementwise add (residual)
        6) LocalResponseNorm
        7) global average pooling over (D, H, W)
        8) final linear projection (fc)

    Args (init):
        in_ch (int): number of input channels
        mid_ch (int): channels after first conv
        out_ch (int): output channels of conv block / input to fc
        kernel_size (int): kernel for first conv (assumed odd, used with padding)
        num_groups (int): groups for GroupNorm (must divide mid_ch)
        dropout_p (float): dropout probability for Dropout3d
        lrn_size (int): LocalResponseNorm window size
        lrn_alpha (float): LRN alpha
        lrn_beta (float): LRN beta
        lrn_k (float): LRN k (bias)
        fc_out (int): dimension of output projection
    """
    def __init__(
        self,
        in_ch: int,
        mid_ch: int,
        out_ch: int,
        kernel_size: int,
        num_groups: int,
        dropout_p: float,
        lrn_size: int,
        lrn_alpha: float,
        lrn_beta: float,
        lrn_k: float,
        fc_out: int,
    ):
        super(Model, self).__init__()
        padding = kernel_size // 2

        # Primary conv block
        self.conv1 = nn.Conv3d(in_ch, mid_ch, kernel_size=kernel_size, padding=padding, bias=False)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=mid_ch)
        self.dropout = nn.Dropout3d(p=dropout_p)
        self.conv2 = nn.Conv3d(mid_ch, out_ch, kernel_size=1, padding=0, bias=False)

        # Skip projection to match channels for residual add
        self.skip_conv = nn.Conv3d(in_ch, out_ch, kernel_size=1, padding=0, bias=False)

        # Local Response Normalization
        self.lrn = nn.LocalResponseNorm(size=lrn_size, alpha=lrn_alpha, beta=lrn_beta, k=lrn_k)

        # Final projection head
        self.fc = nn.Linear(out_ch, fc_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the module.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, D, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (B, fc_out)
        """
        # Conv -> GroupNorm -> ReLU
        out = self.conv1(x)
        out = self.gn(out)
        out = F.relu(out, inplace=True)

        # Spatial-channel dropout (drops entire channels in 3D feature maps)
        out = self.dropout(out)

        # Channel mixing with 1x1x1 conv -> ReLU
        out = self.conv2(out)
        out = F.relu(out, inplace=True)

        # Skip projection and residual addition
        skip = self.skip_conv(x)
        out = out + skip

        # Local response normalization across channels
        out = self.lrn(out)

        # Global average pooling over (D, H, W) -> (B, out_ch)
        out = out.mean(dim=(2, 3, 4))

        # Final projection
        out = self.fc(out)
        return out


def get_inputs():
    """
    Returns a list with a single input tensor matching the configured shapes:
    (batch_size, in_channels, depth, height, width)
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]


def get_init_inputs():
    """
    Returns the initialization parameters in the order expected by Model.__init__
    """
    return [
        in_channels,
        mid_channels,
        out_channels,
        3,              # kernel_size for conv1
        num_groups,
        dropout_p,
        lrn_size,
        lrn_alpha,
        lrn_beta,
        lrn_k,
        fc_out_features,
    ]