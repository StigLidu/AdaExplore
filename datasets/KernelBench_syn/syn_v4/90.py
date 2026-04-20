import torch
import torch.nn as nn
import torch.nn.functional as F

# Configuration
batch_size = 4
in_channels = 3
depth = 4
height = 8
width = 8

# Intermediate/architecture sizes
conv3d_out_channels = 6   # channels after ConvTranspose3d
lp_norm_type = 2          # p for LPPool1d (Euclidean)
lp_kernel = 4             # kernel size for LPPool1d (will downsample by factor ~4)
lazy_conv_out_channels = 4  # out channels for LazyConvTranspose1d

# Final spatial shape after all transforms (must multiply to the final 1D length)
final_d = 8
final_h = 16
final_w = 8
assert final_d * final_h * final_w > 0

class Model(nn.Module):
    """
    Complex model combining ConvTranspose3d, LPPool1d and LazyConvTranspose1d.

    Pipeline:
      1. Upsample 3D volume with ConvTranspose3d (spatial double).
      2. Apply ReLU.
      3. Collapse spatial dims into a 1D sequence (batch, channels, seq_len).
      4. Apply LPPool1d to reduce sequence length.
      5. Apply LazyConvTranspose1d to upsample the sequence length.
      6. Apply a final ReLU and reshape back to a 5D volume (batch, channels, D, H, W).

    Note: The specific kernel/stride choices ensure predictable sequence lengths for
    the fixed input sizes defined at module level.
    """
    def __init__(self):
        super(Model, self).__init__()
        # 3D transposed convolution doubles each spatial dimension (kernel_size=2, stride=2)
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=conv3d_out_channels,
            kernel_size=2,
            stride=2,
            bias=True
        )

        # LPPool1d will perform power-average pooling over the sequence dimension
        # (norm_type p, kernel_size will downsample by ~kernel_size)
        self.lp_pool = nn.LPPool1d(norm_type=lp_norm_type, kernel_size=lp_kernel)

        # LazyConvTranspose1d: we specify only out_channels and kernel/stride.
        # The in_channels will be inferred at first forward pass.
        self.lazy_conv_t1d = nn.LazyConvTranspose1d(
            out_channels=lazy_conv_out_channels,
            kernel_size=2,
            stride=2,
            bias=True
        )

        # Small channel-wise projection after reshaping back to 5D
        self.channel_proj = nn.Conv3d(
            in_channels=lazy_conv_out_channels,
            out_channels=lazy_conv_out_channels,
            kernel_size=1,
            stride=1,
            bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_channels, D, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (batch, lazy_conv_out_channels, final_d, final_h, final_w)
        """
        # 1) Upsample 3D volume
        y = self.conv_transpose3d(x)                 # (B, conv3d_out_channels, D*2, H*2, W*2)
        # 2) Non-linearity
        y = F.relu(y)

        # 3) Collapse spatial dims into a 1D sequence for each channel
        B, C, Dp, Hp, Wp = y.shape
        seq = y.view(B, C, Dp * Hp * Wp)            # (B, C, L1)

        # 4) Apply LPPool1d to reduce sequence length (downsample by ~lp_kernel)
        seq = self.lp_pool(seq)                     # (B, C, L2)

        # 5) Upsample sequence with LazyConvTranspose1d (in_channels inferred at first forward)
        seq = self.lazy_conv_t1d(seq)               # (B, lazy_conv_out_channels, L3)

        # 6) Activation
        seq = F.relu(seq)

        # 7) Reshape sequence back to 5D tensor
        L3 = seq.shape[-1]
        expected = final_d * final_h * final_w
        if L3 != expected:
            # If shapes mismatch, attempt a simple linear projection to resize sequence length.
            # This keeps the model robust to small input size changes.
            seq = seq.permute(0, 2, 1)  # (B, L3, C)
            seq = F.interpolate(seq, size=expected, mode='linear', align_corners=False)
            seq = seq.permute(0, 2, 1)  # (B, C, expected)
            L3 = expected

        out = seq.view(B, lazy_conv_out_channels, final_d, final_h, final_w)  # (B, C_out, Df, Hf, Wf)

        # 8) Small 1x1x1 convolution for channel mixing and final activation
        out = self.channel_proj(out)
        out = F.relu(out)

        return out

def get_inputs():
    """
    Returns a list with a single input tensor appropriate for the Model.
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Returns any initialization inputs required. Lazy layers will infer shapes at first forward,
    so no special initialization values are required.
    """
    return []