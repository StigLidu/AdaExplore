import torch
import torch.nn as nn

class Model(nn.Module):
    """
    3D feature extractor that combines spatial padding, 3D convolution,
    Local Response Normalization across channels, GELU non-linearity,
    and global spatial pooling followed by a channel projection.
    """
    def __init__(
        self,
        in_channels: int,
        conv_out_channels: int,
        out_features: int,
        pad: int = 1,
        lrn_size: int = 5,
    ):
        """
        Args:
            in_channels (int): Number of channels in the input tensor.
            conv_out_channels (int): Number of output channels from the 3D conv.
            out_features (int): Number of features in the final linear projection.
            pad (int): Symmetric padding size applied on each spatial side (D,H,W).
            lrn_size (int): Number of channels to sum over in LocalResponseNorm.
        """
        super(Model, self).__init__()

        # Ensure ZeroPad3d receives a 6-tuple: (Wl, Wr, Hl, Hr, Dl, Dr)
        pad_tuple = (pad, pad, pad, pad, pad, pad)
        self.pad = nn.ZeroPad3d(pad_tuple)

        # 3D convolution: small kernel to process local neighborhoods after explicit padding
        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=conv_out_channels,
            kernel_size=3,
            stride=1,
            padding=0,  # padding already applied via ZeroPad3d
            bias=False
        )

        # Local Response Normalization across channels to encourage lateral inhibition
        self.lrn = nn.LocalResponseNorm(size=lrn_size, alpha=1e-4, beta=0.75, k=2.0)

        # Non-linearity
        self.gelu = nn.GELU()

        # Global spatial pooling to reduce D,H,W -> 1
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Final projection from conv channels -> desired output features
        self.fc = nn.Linear(conv_out_channels, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
         - Zero-pad spatial dimensions
         - 3D convolution over local neighborhoods
         - LocalResponseNorm across channels
         - GELU activation
         - Global average pooling over D,H,W
         - Linear projection to output features

        Args:
            x (torch.Tensor): Input tensor with shape (batch, channels, D, H, W)

        Returns:
            torch.Tensor: Output tensor with shape (batch, out_features)
        """
        x = self.pad(x)           # pad spatial dims
        x = self.conv3d(x)        # conv3d
        x = self.lrn(x)           # local response normalization across channels
        x = self.gelu(x)          # non-linearity
        x = self.pool(x)          # global spatial pooling -> (B, C, 1, 1, 1)
        x = x.view(x.size(0), -1) # flatten channels -> (B, C)
        x = self.fc(x)            # projection -> (B, out_features)
        return x

# Configuration / default sizes
batch_size = 8
in_channels = 32
conv_out_channels = 64
out_features = 128
depth = 16
height = 32
width = 32
pad = 1
lrn_size = 5

def get_inputs():
    """
    Returns example input tensor(s) for the Model forward.
    Shape: (batch_size, in_channels, depth, height, width)
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters for the Model constructor in order:
    [in_channels, conv_out_channels, out_features, pad, lrn_size]
    """
    return [in_channels, conv_out_channels, out_features, pad, lrn_size]