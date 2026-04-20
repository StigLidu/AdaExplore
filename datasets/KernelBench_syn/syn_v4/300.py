import torch
import torch.nn as nn

# Configuration / module-level variables
BATCH_SIZE = 8
IN_CHANNELS = 3
DEPTH = 8
HEIGHT = 16
WIDTH = 32

OUT_CHANNELS = 64
KERNEL_SIZE = 5
NEG_SLOPE = 0.02
# ReflectionPad3d expects a 6-tuple: (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
PAD_3D = (1, 1, 2, 2, 1, 1)


class Model(nn.Module):
    """
    Complex module that:
    - Applies 3D reflection padding to a volumetric input (B, C, D, H, W)
    - Collapses spatial D and H into the channel dimension to form a 1D sequence over W
    - Uses a LazyConv1d to learn a conv over the width dimension (lazy in_channels inferred at first call)
    - Applies a LeakyReLU nonlinearity
    - Computes a channel-wise summary, projects it with a learnable matrix, and uses it to modulate the conv output
    - Reduces the modulated features to produce a per-position output (B, L_out)
    """
    def __init__(self,
                 out_channels: int = OUT_CHANNELS,
                 kernel_size: int = KERNEL_SIZE,
                 neg_slope: float = NEG_SLOPE,
                 pad_3d: tuple = PAD_3D):
        super(Model, self).__init__()
        # 3D reflection padding
        self.pad3d = nn.ReflectionPad3d(pad_3d)
        # Lazy 1D conv: in_channels inferred on first forward call
        # Use padding to preserve sequence length if possible
        self.conv1d = nn.LazyConv1d(out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        # Nonlinear activation
        self.act = nn.LeakyReLU(negative_slope=neg_slope)
        # Learnable projection used to modulate conv channels (channel gating)
        self.register_parameter("channel_proj", nn.Parameter(torch.randn(out_channels, out_channels)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, D, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (B, L_out) where L_out is the sequence length after conv
        """
        # x: (B, C, D, H, W)
        B = x.size(0)

        # 1) Reflection pad the 3D volume
        x_padded = self.pad3d(x)  # (B, C, D_p, H_p, W_p)

        # 2) Collapse channel + depth + height into a single "channel" dimension to form (B, C', W_p)
        B, C, Dp, Hp, Wp = x_padded.shape
        # Merge C, Dp, Hp -> new_channels
        x_seq = x_padded.view(B, C * Dp * Hp, Wp)  # (B, in_channels', seq_len)

        # 3) 1D convolution across the width/sequence dimension
        conv_out = self.conv1d(x_seq)  # (B, out_channels, L_out)

        # 4) Nonlinearity
        conv_act = self.act(conv_out)  # (B, out_channels, L_out)

        # 5) Channel-wise global summary (mean over sequence)
        channel_summary = conv_act.mean(dim=2)  # (B, out_channels)

        # 6) Project the summary with a learnable matrix and use it to modulate conv features
        #    projected: (B, out_channels) @ (out_channels, out_channels) -> (B, out_channels)
        projected = torch.matmul(channel_summary, self.channel_proj)  # (B, out_channels)
        gate = projected.unsqueeze(2)  # (B, out_channels, 1)

        # 7) Modulate conv features and reduce across channels to produce final per-position outputs
        gated = conv_act * gate  # (B, out_channels, L_out)
        out = gated.sum(dim=1)  # (B, L_out)

        return out


def get_inputs():
    """
    Generate input tensors for testing the Model.

    Returns:
        list: [x] where x has shape (BATCH_SIZE, IN_CHANNELS, DEPTH, HEIGHT, WIDTH)
    """
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, DEPTH, HEIGHT, WIDTH)
    return [x]


def get_init_inputs():
    """
    Return initialization parameters that can be used to construct the Model externally.

    Returns:
        list: [out_channels, kernel_size, neg_slope, pad_3d]
    """
    return [OUT_CHANNELS, KERNEL_SIZE, NEG_SLOPE, PAD_3D]