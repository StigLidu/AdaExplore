import torch
import torch.nn as nn

# Module-level configuration variables
BATCH = 4
IN_CHANNELS = 3
HEIGHT = 128
WIDTH = 128
DOWNSCALE = 2            # PixelUnshuffle downscale factor
OUT_CHANNELS = 64        # Output channel count after linear projection

class Model(nn.Module):
    """
    A moderately complex module that:
      - Uses PixelUnshuffle to reduce spatial resolution while increasing channels.
      - Centers each channel by subtracting its spatial mean.
      - Applies a clipped ReLU (ReLU6) non-linearity.
      - Projects per-pixel channel vectors using a LazyLinear (in_features inferred at first forward).
      - Applies a learned channel mixing matrix (provided as input) via einsum.
      - Adds a global residual derived from the spatial means projected into output channels.
      - Final non-linearity (ReLU6) applied to the sum.
    This combines spatial re-arrangement, channel-wise statistics, lazy-initialized linear layers,
    and an external channel mixing matrix to produce a transformed feature map.
    """
    def __init__(self, out_channels: int = OUT_CHANNELS, downscale: int = DOWNSCALE):
        super(Model, self).__init__()
        # PixelUnshuffle will transform (N, C, H, W) -> (N, C * r^2, H/r, W/r)
        self.unshuffle = nn.PixelUnshuffle(downscale)
        self.relu6 = nn.ReLU6()
        # LazyLinear will infer in_features at first forward call.
        # It maps channel-vectors at each spatial location to out_channels.
        self.linear = nn.LazyLinear(out_channels)
        self.out_channels = out_channels
        self.downscale = downscale

    def forward(self, x: torch.Tensor, channel_mixer: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image tensor of shape (N, C, H, W)
            channel_mixer: Square mixing matrix of shape (out_channels, out_channels)
                            that will be used to mix output channels per spatial location.

        Returns:
            Tensor of shape (N, out_channels, H/downscale, W/downscale)
        """
        # 1) Rearrange spatial information into channels
        x_un = self.unshuffle(x)  # (N, C * r^2, H/r, W/r)

        # 2) Center channels by subtracting their spatial mean (per-sample, per-channel)
        mean = x_un.mean(dim=(2, 3), keepdim=True)  # (N, C', 1, 1)
        x_centered = x_un - mean

        # 3) Non-linearity
        x_act = self.relu6(x_centered)

        # 4) Prepare per-pixel channel vectors for a linear projection
        #    Move channels to last dim and flatten spatial positions into batch dimension
        N, Cprime, H2, W2 = x_act.shape
        x_flat = x_act.permute(0, 2, 3, 1).reshape(-1, Cprime)  # (N * H2 * W2, C')

        # 5) Project channel-vectors to out_channels using LazyLinear (in_features inferred here)
        x_proj = self.linear(x_flat)  # (N * H2 * W2, out_channels)

        # 6) Reshape back to (N, out_channels, H2, W2)
        x_proj_reshaped = x_proj.view(N, H2, W2, self.out_channels).permute(0, 3, 1, 2)

        # 7) Mix channels with the provided channel_mixer matrix
        #    Uses einsum to compute per-location channel mixing: (b,c,h,w) x (c,d) -> (b,d,h,w)
        x_mixed = torch.einsum('bchw,cd->bdhw', x_proj_reshaped, channel_mixer)

        # 8) Compute a global residual: project the spatial means into output channels and add
        #    Use the same LazyLinear on the squeezed mean (shape: (N, C')) -> (N, out_channels)
        mean_vec = mean.squeeze(-1).squeeze(-1)      # (N, C')
        mean_proj = self.linear(mean_vec)            # (N, out_channels)
        mean_proj = mean_proj.unsqueeze(-1).unsqueeze(-1)  # (N, out_channels, 1, 1)

        out = x_mixed + mean_proj  # broadcast add

        # 9) Final activation
        out = self.relu6(out)

        return out

# Functions expected by the harness / tests

def get_inputs():
    """
    Generates:
      - An input image tensor of shape (BATCH, IN_CHANNELS, HEIGHT, WIDTH)
      - A symmetric channel mixing matrix of shape (OUT_CHANNELS, OUT_CHANNELS)
    """
    x = torch.randn(BATCH, IN_CHANNELS, HEIGHT, WIDTH)

    # Create a symmetric mixing matrix for stability / interpretability
    M = torch.randn(OUT_CHANNELS, OUT_CHANNELS)
    M = (M + M.T) / 2.0
    return [x, M]

def get_init_inputs():
    """
    Returns initialization inputs for constructing the Model: [out_channels, downscale]
    """
    return [OUT_CHANNELS, DOWNSCALE]