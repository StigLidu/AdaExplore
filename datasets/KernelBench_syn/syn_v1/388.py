import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex hybrid 2D/1D module that:
      - Applies a strided 2D convolution to downsample spatial resolution by 2
      - Applies a second 2D convolution + ReLU to extract features
      - Treats the width dimension as a sequence and upsamples it using ConvTranspose1d
      - Restores spatial layout and upsamples the height by integer repetition to recover original size
      - Applies a final 1x1 convolution to project to the desired output channels and a final ReLU

    This demonstrates a mixed use of nn.Conv2d, nn.ConvTranspose1d and nn.ReLU,
    together with careful tensor reshaping/permutes.
    """
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int):
        super(Model, self).__init__()
        # Downsampling conv: halves H and W (stride=2, kernel=3, padding=1)
        self.conv_down = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=2, padding=1, bias=True)
        # Feature extractor (keeps the downsampled spatial size)
        self.conv_feat = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=True)
        # 1D transposed convolution to upsample the width dimension by factor 2
        # It operates on sequences of length W//2 and produces length W. Input channels = mid_channels.
        self.deconv_width = nn.ConvTranspose1d(mid_channels, mid_channels, kernel_size=2, stride=2, padding=0, bias=True)
        # Final 1x1 conv to project mid_channels -> out_channels
        self.conv_proj = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        # Non-linearity
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C_in, H, W). H and W must be divisible by 2.

        Returns:
            Tensor of shape (B, C_out, H, W) (restored to original spatial size)
        """
        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0, "H and W must be divisible by 2"

        # Step 1: Downsample spatially by 2x using conv2d
        y = self.conv_down(x)                 # (B, mid, H//2, W//2)

        # Step 2: Feature extraction
        y = self.conv_feat(y)                 # (B, mid, H//2, W//2)
        y = self.relu(y)

        # Save sizes
        B, mid, H2, W2 = y.shape              # H2 = H//2, W2 = W//2

        # Step 3: Treat width as sequence and upsample width using ConvTranspose1d
        # Prepare for ConvTranspose1d: (N_seq, C_in, L) where N_seq = B * H2
        # Permute to (B, H2, mid, W2) -> reshape to (B*H2, mid, W2)
        y_perm = y.permute(0, 2, 1, 3).contiguous()    # (B, H2, mid, W2)
        y_seq = y_perm.view(B * H2, mid, W2)           # (B*H2, mid, W2)

        # Apply 1D transposed conv to upsample width by factor 2 -> length becomes W2*2 == W
        y_up_w = self.deconv_width(y_seq)              # (B*H2, mid, W)

        # Restore back to 4D: (B, H2, mid, W) -> permute to (B, mid, H2, W)
        y_up_w = y_up_w.view(B, H2, mid, -1).permute(0, 2, 1, 3).contiguous()  # (B, mid, H2, W)

        # Step 4: Project channels to desired out_channels using 1x1 conv
        y_proj = self.conv_proj(y_up_w)    # (B, out_channels, H2, W)

        # Step 5: Upsample height by integer repetition to recover H (H2 * 2 == H)
        y_out = torch.repeat_interleave(y_proj, repeats=2, dim=2)  # (B, out_channels, H, W)

        # Final non-linearity
        y_out = self.relu(y_out)

        return y_out

# Configuration / typical sizes
batch_size = 8
in_channels = 3
mid_channels = 32
out_channels = 16
height = 64  # must be divisible by 2
width = 64   # must be divisible by 2

def get_inputs():
    """
    Returns:
        [x]: list containing the input tensor with shape (batch_size, in_channels, height, width)
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters required to construct the Model:
        [in_channels, mid_channels, out_channels]
    """
    return [in_channels, mid_channels, out_channels]