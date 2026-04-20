import torch
import torch.nn as nn
from typing import Tuple, List

# Configuration (module-level)
batch_size = 8
in_channels_3d = 16
in_channels_2d = 8
depth = 4
height = 64
width = 64

# Adaptive pooling will reduce spatial HxW from (height, width) -> (pooled_h, pooled_w)
pooled_h = 32
pooled_w = 32

# Fractional pool settings (will reduce HxW by ~50%)
frac_kernel = (2, 2)
frac_output_ratio = (0.5, 0.5)

# ConvTranspose1d settings (will expand the flattened spatial length by factor ~2)
conv_out_channels = 12
conv_kernel_size = 4
conv_stride = 2
conv_padding = 1

class Model(nn.Module):
    """
    Complex module that mixes 3D adaptive pooling, 2D fractional max pooling, and a 1D transposed convolution.

    Computation flow:
    1. AdaptiveMaxPool3d reduces (D, H, W) -> (1, pooled_h, pooled_w).
    2. Squeeze depth -> produce a 4D tensor compatible with the 2D pooled tensor.
    3. FractionalMaxPool2d reduces a separate 4D input's spatial dims by output_ratio.
    4. Concatenate along channel dimension.
    5. Flatten spatial dims into a sequence length and apply ConvTranspose1d to upsample the sequence.
    6. Reshape back into a 4D tensor with expanded spatial dimensions.
    """
    def __init__(
        self,
        in_channels_3d: int,
        in_channels_2d: int,
        pooled_hw: Tuple[int, int],
        frac_kernel: Tuple[int, int],
        frac_output_ratio: Tuple[float, float],
        conv_out_channels: int,
        conv_kernel_size: int,
        conv_stride: int,
        conv_padding: int,
    ):
        super(Model, self).__init__()

        self.in_c3 = in_channels_3d
        self.in_c2 = in_channels_2d
        self.pooled_h, self.pooled_w = pooled_hw

        # Adaptive pool to collapse depth to 1 and spatially pool to pooled_h x pooled_w
        self.adapt_pool = nn.AdaptiveMaxPool3d((1, self.pooled_h, self.pooled_w))

        # Fractional max pool to reduce the other 2D input by roughly frac_output_ratio
        # kernel_size must be provided; output_ratio controls output size
        self.frac_pool = nn.FractionalMaxPool2d(kernel_size=frac_kernel, output_ratio=frac_output_ratio)

        # ConvTranspose1d will take flattened (H'*W') sequence and upsample it.
        # Input channels equal to concatenated channels after pooling.
        concat_channels = self.in_c3 + self.in_c2
        self.conv_t1d = nn.ConvTranspose1d(
            in_channels=concat_channels,
            out_channels=conv_out_channels,
            kernel_size=conv_kernel_size,
            stride=conv_stride,
            padding=conv_padding
        )

        # A small activation for non-linearity
        self.act = nn.ReLU()

    def forward(self, x3d: torch.Tensor, x4d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x3d: Tensor of shape (N, C3, D, H, W)
            x4d: Tensor of shape (N, C2, H, W)

        Returns:
            out: Tensor of shape (N, conv_out_channels, H_out, W_out)
        """
        # Validate input shapes minimally
        assert x3d.dim() == 5, f"x3d must be 5D (N,C,D,H,W), got {x3d.shape}"
        assert x4d.dim() == 4, f"x4d must be 4D (N,C,H,W), got {x4d.shape}"
        N = x3d.size(0)
        # 1) AdaptiveMaxPool3d -> shape (N, C3, 1, pooled_h, pooled_w)
        y3 = self.adapt_pool(x3d)

        # 2) Remove the depth dimension -> (N, C3, pooled_h, pooled_w)
        y3_4d = y3.squeeze(2)  # remove size-1 depth

        # 3) FractionalMaxPool2d on the 4D input -> reduces spatial dims
        # FractionalMaxPool2d expects (N, C, H, W)
        y4 = self.frac_pool(x4d)

        # 4) Ensure spatial dimensions match between y3_4d and y4
        # If shapes don't match, try a center-crop on the larger one (simple, deterministic)
        ph, pw = y3_4d.shape[2], y3_4d.shape[3]
        h4, w4 = y4.shape[2], y4.shape[3]
        if (ph, pw) != (h4, w4):
            # perform center crop on the larger tensor to match the smaller dims
            target_h = min(ph, h4)
            target_w = min(pw, w4)

            def center_crop(t: torch.Tensor, th: int, tw: int) -> torch.Tensor:
                _, _, H, W = t.shape
                start_h = (H - th) // 2
                start_w = (W - tw) // 2
                return t[:, :, start_h:start_h + th, start_w:start_w + tw]

            y3_4d = center_crop(y3_4d, target_h, target_w)
            y4 = center_crop(y4, target_h, target_w)

        # 5) Concatenate along channel dimension -> (N, C3 + C2, H', W')
        concat = torch.cat([y3_4d, y4], dim=1)

        # 6) Flatten spatial dims to a sequence for ConvTranspose1d
        N, C_concat, Hs, Ws = concat.shape
        seq_len = Hs * Ws
        seq = concat.view(N, C_concat, seq_len)  # (N, C_concat, L)

        # 7) Apply ConvTranspose1d to upsample the sequence length (L -> ~2*L)
        seq_up = self.conv_t1d(seq)  # (N, conv_out_channels, L_up)
        seq_up = self.act(seq_up)

        # 8) Reshape back to 4D. Choose H_out and W_out such that H_out * W_out == L_up.
        L_up = seq_up.size(2)

        # For determinism, try to reconstruct into (height, width//2) style if possible.
        # We'll attempt to set H_out = 2 * Hs (i.e., double height), W_out = Ws
        candidate_H = Hs * 2
        candidate_W = Ws
        if candidate_H * candidate_W != L_up:
            # fallback: try to keep the original height (Hs) and double width
            candidate_H = Hs
            candidate_W = Ws * 2
            if candidate_H * candidate_W != L_up:
                # final fallback: set H_out = original height and compute W_out via integer division
                candidate_H = Hs
                candidate_W = L_up // candidate_H
                # If not divisible, last option is to pad/truncate to make divisible.
                if candidate_H * candidate_W != L_up:
                    # truncate or pad sequence to fit exactly candidate_H * candidate_W
                    needed = candidate_H * candidate_W
                    if needed < L_up:
                        seq_up = seq_up[:, :, :needed]
                    else:
                        pad_amount = needed - L_up
                        seq_up = torch.cat([seq_up, torch.zeros((N, seq_up.size(1), pad_amount), dtype=seq_up.dtype, device=seq_up.device)], dim=2)
                    L_up = seq_up.size(2)
        H_out, W_out = candidate_H, candidate_W

        out = seq_up.view(N, seq_up.size(1), H_out, W_out)
        return out

def get_inputs() -> List[torch.Tensor]:
    """
    Returns example inputs:
    - x3d: (batch_size, in_channels_3d, depth, height, width)
    - x4d: (batch_size, in_channels_2d, height, width)
    """
    x3d = torch.randn(batch_size, in_channels_3d, depth, height, width, dtype=torch.float32)
    x4d = torch.randn(batch_size, in_channels_2d, height, width, dtype=torch.float32)
    return [x3d, x4d]

def get_init_inputs() -> List:
    """
    Returns the initialization parameters for Model constructor:
    (in_channels_3d, in_channels_2d, (pooled_h, pooled_w), frac_kernel, frac_output_ratio,
     conv_out_channels, conv_kernel_size, conv_stride, conv_padding)
    """
    return [
        in_channels_3d,
        in_channels_2d,
        (pooled_h, pooled_w),
        frac_kernel,
        frac_output_ratio,
        conv_out_channels,
        conv_kernel_size,
        conv_stride,
        conv_padding,
    ]