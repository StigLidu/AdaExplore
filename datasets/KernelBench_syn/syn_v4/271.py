import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class Model(nn.Module):
    """
    Complex pooling and channel-gating model that combines Lp pooling, 1x1 convolution,
    adaptive average pooling and 1D average pooling to produce a spatial refinement
    which is then used to spatially recalibrate the original input.

    Computation pattern (high level):
      1. Lp pooling (nn.LPPool2d) to perform a power-average downsampling.
      2. 1x1 convolution to mix channel information.
      3. AdaptiveAvgPool2d to a small spatial grid (Oh, Ow).
      4. Reshape to apply AvgPool1d across the width dimension after folding rows into channels.
      5. Compute channel-wise gating (using mean pooling + a small linear layer with sigmoid).
      6. Use the gating to scale the original input channels and add an upsampled refined map.

    This pattern exercises multiple pooling primitives and shape manipulations
    to achieve a non-trivial dataflow while remaining deterministic and fully
    compatible with PyTorch autograd.
    """
    def __init__(
        self,
        in_channels: int,
        lp_p: int,
        lp_kernel: int,
        adaptive_out: Tuple[int, int],
        avg1d_kernel: int
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            lp_p (int): The p-norm degree for LPPool2d.
            lp_kernel (int): Kernel size (and stride) for LPPool2d.
            adaptive_out (Tuple[int,int]): Output (H, W) size for AdaptiveAvgPool2d.
            avg1d_kernel (int): Kernel size (and stride) for AvgPool1d applied across width.
        """
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.lp_p = lp_p
        self.lp_kernel = lp_kernel
        self.adaptive_out = adaptive_out
        self.avg1d_kernel = avg1d_kernel

        # Lp pooling to perform a power-average spatial downsampling
        self.lppool = nn.LPPool2d(norm_type=self.lp_p, kernel_size=self.lp_kernel, stride=self.lp_kernel)

        # Simple channel mixing via pointwise convolution (no spatial extent)
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

        # Adaptive average pool to a compact spatial grid (Oh, Ow)
        self.adapt_pool = nn.AdaptiveAvgPool2d(self.adaptive_out)

        # 1D average pooling will be applied across the width dimension after reshaping.
        self.avg1d = nn.AvgPool1d(kernel_size=self.avg1d_kernel, stride=self.avg1d_kernel)

        # Small fully-connected layer to produce channel gating coefficients
        self.fc_gating = nn.Linear(in_channels, in_channels)

        # Initialize weights sensibly
        nn.init.kaiming_uniform_(self.conv1x1.weight, a=0.01)
        nn.init.xavier_uniform_(self.fc_gating.weight)
        if self.fc_gating.bias is not None:
            nn.init.zeros_(self.fc_gating.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Output tensor of same spatial shape as input (B, C, H, W)
        """
        B, C, H, W = x.shape
        # 1) Lp pooling -> downsample spatially
        y = self.lppool(x)  # (B, C, H1, W1)

        # 2) 1x1 conv to mix channels (keeps spatial dims)
        y = self.conv1x1(y)  # (B, C, H1, W1)

        # 3) Adaptive average pool to compact grid (Oh, Ow)
        y = self.adapt_pool(y)  # (B, C, Oh, Ow)
        Oh, Ow = self.adaptive_out

        # 4) Fold rows into channels and apply AvgPool1d across width to further compress width
        #    Reshape: (B, C, Oh, Ow) -> (B, C*Oh, Ow) to treat each original row-channel as a channel sequence
        y_fold = y.view(B, C * Oh, Ow)  # (B, C*Oh, Ow)
        y_pooled = self.avg1d(y_fold)  # (B, C*Oh, Ow2)
        Ow2 = y_pooled.size(-1)

        # 5) Reshape back to 4D refined map: (B, C, Oh, Ow2)
        refined = y_pooled.view(B, C, Oh, Ow2)  # (B, C, Oh, Ow2)

        # 6) Channel-wise descriptor computed from refined map
        channel_desc = refined.mean(dim=(2, 3))  # (B, C)

        # 7) Gating: small FC + sigmoid to produce per-channel weights in (0,1)
        gating = torch.sigmoid(self.fc_gating(channel_desc))  # (B, C)

        # 8) Scale original input channels
        x_scaled = x * gating.view(B, C, 1, 1)  # (B, C, H, W)

        # 9) Upsample the refined map back to input spatial resolution and combine
        refined_upsampled = F.interpolate(refined, size=(H, W), mode='bilinear', align_corners=False)  # (B, C, H, W)

        # 10) Final combination (residual refinement)
        out = x_scaled + 0.5 * refined_upsampled

        return out

# Configuration variables (module-level)
BATCH_SIZE = 8
IN_CHANNELS = 64
HEIGHT = 128
WIDTH = 128

LP_P = 2              # L2 pooling
LP_KERNEL = 4         # spatial downsample factor for Lp pooling
ADAPTIVE_OUT = (8, 8) # target grid from adaptive pooling
AVG1D_KERNEL = 2      # 1D pooling across width after folding rows into channels

def get_inputs():
    """
    Returns example input tensors for the model.

    Output:
        [x] where x is a float tensor of shape (BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)
    """
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters matching Model.__init__ signature:
      (in_channels, lp_p, lp_kernel, adaptive_out, avg1d_kernel)
    """
    return [IN_CHANNELS, LP_P, LP_KERNEL, ADAPTIVE_OUT, AVG1D_KERNEL]