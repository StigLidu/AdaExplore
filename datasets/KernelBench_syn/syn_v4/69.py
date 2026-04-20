import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

class Model(nn.Module):
    """
    Composite model that demonstrates a mixed 1D/3D processing pipeline:
    - A 1D path that is Zero-padded then upsampled with a LazyConvTranspose1d.
    - A 3D path that is projected (Conv3d), max-pooled (producing indices) and then
      partially inverted with MaxUnpool3d.
    - The recovered 3D features are spatially aggregated and used to modulate
      the 1D upsampled signal (elementwise scaling), followed by a final non-linearity.

    This creates a cross-domain gating mechanism where volumetric context
    influences a sequence upsampling operation.
    """
    def __init__(
        self,
        conv_t_kernel: int,
        conv_t_stride: int,
        pad: Tuple[int, int],
        pool_kernel: int,
        pool_stride: int,
        conv3d_out_channels: int,
    ):
        """
        Initializes the composite model.

        Args:
            conv_t_kernel (int): Kernel size for ConvTranspose1d.
            conv_t_stride (int): Stride for ConvTranspose1d.
            pad (Tuple[int, int]): (left, right) padding for ZeroPad1d on the sequence.
            pool_kernel (int): Kernel size for MaxPool3d.
            pool_stride (int): Stride for MaxPool3d.
            conv3d_out_channels (int): Output channels for the 3D projection conv.
        """
        super(Model, self).__init__()

        # Sequence path components
        # LazyConvTranspose1d will infer in_channels at first forward pass.
        self.pad = nn.ZeroPad1d(pad)
        self.deconv = nn.LazyConvTranspose1d(
            out_channels=conv3d_out_channels,
            kernel_size=conv_t_kernel,
            stride=conv_t_stride,
            padding=0  # we control padding via ZeroPad1d
        )

        # 3D path components
        # Project input 3D channels to match deconv out channels for easy fusion.
        self.conv3d_proj = nn.Conv3d(in_channels=channels_3d_in, out_channels=conv3d_out_channels, kernel_size=1)
        self.pool3d = nn.MaxPool3d(kernel_size=pool_kernel, stride=pool_stride, return_indices=True)
        self.unpool3d = nn.MaxUnpool3d(kernel_size=pool_kernel, stride=pool_stride)

        # A small refinement after fusion
        self.post_act = nn.ReLU(inplace=True)

    def forward(self, seq: torch.Tensor, vol: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining 1D and 3D information.

        Args:
            seq (torch.Tensor): Sequence tensor of shape (batch, seq_in_channels, seq_len).
            vol (torch.Tensor): Volumetric tensor of shape (batch, channels_3d_in, D, H, W).

        Returns:
            torch.Tensor: Fused sequence tensor (batch, out_channels, seq_len_up).
        """
        # 1) Zero-pad the sequence on the last dimension
        seq_padded = self.pad(seq)  # shape: (B, C_seq, L_padded)

        # 2) Upsample / transpose-convolution the sequence
        seq_up = self.deconv(seq_padded)  # shape: (B, out_ch, L_up)

        # 3) Project the 3D volume to the same channel dimensionality
        vol_proj = self.conv3d_proj(vol)  # shape: (B, out_ch, D, H, W)

        # 4) MaxPool3d with indices (we will invert it with MaxUnpool3d)
        pooled, indices = self.pool3d(vol_proj)  # pooled: (B, out_ch, Dp, Hp, Wp)

        # 5) Attempt to recover spatial structure via unpooling
        # Provide output_size to ensure exact reconstruction shape when possible
        try:
            unpooled = self.unpool3d(pooled, indices, output_size=vol_proj.size())
        except Exception:
            # Fallback: if output_size mismatch for any reason, call without output_size
            unpooled = self.unpool3d(pooled, indices)

        # 6) Aggregate volumetric features into a per-channel gating vector
        # Use global average over spatial dims to produce (B, out_ch)
        gating = unpooled.mean(dim=(2, 3, 4))  # shape: (B, out_ch)

        # 7) Broadcast gating to sequence length and modulate the upsampled sequence
        # gating -> (B, out_ch, 1) -> expand to (B, out_ch, L_up)
        gating_expanded = gating.unsqueeze(-1).expand(-1, -1, seq_up.size(2))
        fused = seq_up * gating_expanded  # elementwise modulation

        # 8) Final activation / refinement
        out = self.post_act(fused)

        return out


# ---------------------------
# Module-level configuration
# ---------------------------
batch_size = 4

# Sequence path configuration
seq_in_channels = 3
seq_len = 128
conv_t_kernel = 4
conv_t_stride = 2
pad = (1, 2)  # (left, right) padding for ZeroPad1d
deconv_out_channels = 8  # matches the 3D projection out channels

# Volumetric path configuration
channels_3d_in = 6
depth = 16
height = 16
width = 8
pool_kernel = 2
pool_stride = 2

def get_inputs() -> List[torch.Tensor]:
    """
    Builds example input tensors for the model:
    - seq: 1D sequence tensor (batch, seq_in_channels, seq_len)
    - vol: 3D volumetric tensor (batch, channels_3d_in, D, H, W)
    """
    seq = torch.randn(batch_size, seq_in_channels, seq_len)
    vol = torch.randn(batch_size, channels_3d_in, depth, height, width)
    return [seq, vol]

def get_init_inputs() -> List:
    """
    Returns the initialization parameters for Model in the same order as __init__:
    (conv_t_kernel, conv_t_stride, pad, pool_kernel, pool_stride, conv3d_out_channels)
    """
    return [conv_t_kernel, conv_t_stride, pad, pool_kernel, pool_stride, deconv_out_channels]