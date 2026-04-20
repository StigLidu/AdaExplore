import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    A moderately complex vision-style module that demonstrates a sequence of:
      - 1x1 convolution to project input channels
      - Lp pooling to aggregate local neighborhoods
      - a 3x3 convolution for local mixing
      - Tanh non-linearity
      - FeatureAlphaDropout for channel-wise stochastic regularization
      - global spatial pooling and a final linear layer for output logits

    This combines nn.LPPool2d, nn.Tanh, and nn.FeatureAlphaDropout in a small pipeline.
    """
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_features: int,
        p: int = 2,
        lp_kernel: int = 2,
        dropout_prob: float = 0.15
    ):
        """
        Initializes the module.

        Args:
            in_channels (int): Number of input channels.
            mid_channels (int): Number of intermediate channels after projection.
            out_features (int): Size of the final linear output.
            p (int, optional): The norm degree for LPPool2d. Defaults to 2.
            lp_kernel (int, optional): Kernel size (and stride) for LPPool2d. Defaults to 2.
            dropout_prob (float, optional): Drop probability for FeatureAlphaDropout. Defaults to 0.15.
        """
        super(Model, self).__init__()
        # Project input channels to a working channel dimension
        self.project = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=True)
        # Lp pooling to perform a power-mean pooling over local windows
        self.lp_pool = nn.LPPool2d(norm_type=p, kernel_size=lp_kernel, stride=lp_kernel)
        # 3x3 conv to mix spatial neighborhoods (keeps mid_channels)
        self.local_mix = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=True)
        # Non-linearity
        self.tanh = nn.Tanh()
        # Channel-wise dropout appropriate for SELU-like setups / feature regularization
        self.feat_dropout = nn.FeatureAlphaDropout(p=dropout_prob)
        # Global pooling to produce a per-channel descriptor
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Final linear classifier / regressor head
        self.fc = nn.Linear(mid_channels, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_channels, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (batch, out_features)
        """
        # Project channels
        x = self.project(x)                            # (B, mid_channels, H, W)
        # Apply Lp pooling to reduce spatial resolution while using Lp norm
        x = self.lp_pool(x)                            # (B, mid_channels, H//k, W//k)
        # Local mixing with a 3x3 convolution
        x = self.local_mix(x)                          # (B, mid_channels, H//k, W//k)
        # Non-linearity
        x = self.tanh(x)                               # (B, mid_channels, H//k, W//k)
        # Feature-wise dropout for regularization (channel dropout)
        x = self.feat_dropout(x)                       # (B, mid_channels, H//k, W//k)
        # Global pooling to collapse spatial dims
        x = self.global_pool(x)                        # (B, mid_channels, 1, 1)
        x = x.view(x.size(0), -1)                      # (B, mid_channels)
        # Final linear projection
        out = self.fc(x)                               # (B, out_features)
        return out

# Configuration / default sizes
batch_size = 8
in_channels = 3
height = 128
width = 128
mid_channels = 64
out_features = 10
lp_p = 2
lp_kernel = 2
dropout_prob = 0.15

def get_inputs():
    """
    Returns typical input tensors for running the model's forward pass.
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters matching Model.__init__ signature.
    """
    return [in_channels, mid_channels, out_features, lp_p, lp_kernel, dropout_prob]