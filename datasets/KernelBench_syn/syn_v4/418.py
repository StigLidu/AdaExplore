import torch
import torch.nn as nn

# Configuration variables
batch_size = 8
in_channels = 3
height = 32
width = 32
downscale = 2  # PixelUnshuffle downscale factor
lp_p = 2       # LPPool p-norm
lp_kernel = 3  # LPPool kernel size
lp_stride = 2  # LPPool stride
fc_out_features = 64

class Model(nn.Module):
    """
    Complex model combining PixelUnshuffle, LPPool1d, SELU activation, and a final linear projection.

    Computation pipeline:
    1. PixelUnshuffle with a configurable downscale factor. (N, C, H, W) -> (N, C*r^2, H/r, W/r)
    2. Flatten spatial dimensions into a length dimension to form (N, C', L).
    3. Apply LPPool1d across the length dimension to produce a pooled sequence.
    4. Apply SELU activation element-wise.
    5. Global average across the length dimension to get a channel vector (N, C').
    6. Linear projection to an output feature vector and final SELU activation.
    """
    def __init__(
        self,
        in_channels: int = in_channels,
        downscale: int = downscale,
        lp_p: int = lp_p,
        lp_kernel: int = lp_kernel,
        lp_stride: int = lp_stride,
        fc_out: int = fc_out_features
    ):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.downscale = downscale

        # Moves blocks of spatial pixels into channel dimension
        self.pixel_unshuffle = nn.PixelUnshuffle(self.downscale)

        # After unshuffle, channels become in_channels * downscale^2
        self.post_channels = self.in_channels * (self.downscale ** 2)

        # 1D power-average pooling over the flattened spatial length dimension
        # p-norm pooling with configurable kernel and stride
        self.lppool = nn.LPPool1d(lp_p, kernel_size=lp_kernel, stride=lp_stride)

        # SELU activation for self-normalizing behavior
        self.selu = nn.SELU()

        # Final linear projection from per-channel pooled summary to feature vector
        self.fc = nn.Linear(self.post_channels, fc_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (N, fc_out)
        """
        # 1) PixelUnshuffle: (N, C, H, W) -> (N, C * r^2, H/r, W/r)
        x_un = self.pixel_unshuffle(x)

        # 2) Flatten spatial dimensions into a length dimension: (N, C', H', W') -> (N, C', L)
        n, c_p, h_p, w_p = x_un.shape
        x_flat = x_un.view(n, c_p, h_p * w_p)

        # 3) Apply LPPool1d along L dimension -> reduces length
        x_pooled = self.lppool(x_flat)

        # 4) SELU activation
        x_activated = self.selu(x_pooled)

        # 5) Global average across the length dimension to get per-channel summary (N, C')
        x_summary = x_activated.mean(dim=2)

        # 6) Final linear projection and SELU
        out = self.fc(x_summary)
        out = self.selu(out)

        return out

def get_inputs():
    """
    Returns a list containing a single input tensor for the model:
    Shape: (batch_size, in_channels, height, width)
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters for the Model constructor in the same order:
    in_channels, downscale, lp_p, lp_kernel, lp_stride, fc_out
    """
    return [in_channels, downscale, lp_p, lp_kernel, lp_stride, fc_out_features]