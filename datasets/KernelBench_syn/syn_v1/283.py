import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex module that fuses a volumetric branch (3D padded -> spatial pooling -> ConvTranspose1d)
    with a sequence branch (Conv1d compatible). The volumetric branch is zero-padded in 3D,
    collapsed over spatial H/W to a 1D sequence over depth, processed with a ConvTranspose1d,
    activated with RReLU, then concatenated with the input sequence and mixed with a 1x1 Conv1d.

    Forward signature:
        forward(seq: Tensor[B, C_seq, L], vol: Tensor[B, C_vol, D, H, W]) -> Tensor[B, out_channels, L]
    """
    def __init__(self,
                 seq_channels: int,
                 vol_channels: int,
                 mid_channels: int,
                 out_channels: int,
                 deconv_kernel: int = 3,
                 pad_3d=(0, 0, 0, 0, 1, 1)):
        """
        Args:
            seq_channels: channels of the input sequence tensor
            vol_channels: channels of the volumetric input
            mid_channels: channels produced by ConvTranspose1d from volumetric branch
            out_channels: final output channels after mixing
            deconv_kernel: kernel size for ConvTranspose1d (odd preferred to preserve lengths)
            pad_3d: tuple for ZeroPad3d (left, right, top, bottom, front, back)
        """
        super(Model, self).__init__()
        self.seq_channels = seq_channels
        self.vol_channels = vol_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels

        # ZeroPad3d for volumetric branch
        self.pad3d = nn.ZeroPad3d(pad_3d)

        # ConvTranspose1d to transform depth-dimension as a sequence
        # Keep stride=1 and padding to preserve length relative to padded depth
        deconv_padding = (deconv_kernel - 1) // 2
        self.deconv1d = nn.ConvTranspose1d(
            in_channels=vol_channels,
            out_channels=mid_channels,
            kernel_size=deconv_kernel,
            stride=1,
            padding=deconv_padding
        )

        # Randomized leaky ReLU activation
        self.act = nn.RReLU(lower=0.125, upper=0.333, inplace=False)

        # 1x1 Conv1d to mix concatenated channels from seq and volumetric branches
        self.mix_conv = nn.Conv1d(
            in_channels=seq_channels + mid_channels,
            out_channels=out_channels,
            kernel_size=1
        )

        # Optional small dropout to add a bit of regularization
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, seq: torch.Tensor, vol: torch.Tensor) -> torch.Tensor:
        """
        seq: Tensor[B, C_seq, L]
        vol: Tensor[B, C_vol, D, H, W]
        returns: Tensor[B, out_channels, L]
        """
        # 1) Zero-pad the volumetric input (B, C_vol, D_p, H_p, W_p)
        vol_padded = self.pad3d(vol)

        # 2) Collapse spatial H and W with mean to get (B, C_vol, D_p)
        vol_collapsed = vol_padded.mean(dim=(3, 4))  # mean over H and W

        # 3) ConvTranspose1d over the depth-as-sequence dimension
        #    vol_collapsed has shape (B, C_vol, L_v) compatible with ConvTranspose1d
        vol_deconv = self.deconv1d(vol_collapsed)

        # 4) Apply randomized leaky ReLU activation
        vol_act = self.act(vol_deconv)

        # 5) Concatenate sequence branch and volumetric branch along the channel dim
        # Ensure lengths match: seq shape (B, C_seq, L), vol_act (B, mid_channels, L)
        if seq.size(2) != vol_act.size(2):
            # If lengths mismatch, align by trimming or padding vol_act to seq length
            target_len = seq.size(2)
            cur_len = vol_act.size(2)
            if cur_len > target_len:
                vol_act = vol_act[..., :target_len]
            else:
                # pad last dimension (sequence length) with zeros to match
                pad_amount = target_len - cur_len
                vol_act = nn.functional.pad(vol_act, (0, pad_amount))

        merged = torch.cat([seq, vol_act], dim=1)  # (B, C_seq + mid_channels, L)

        # 6) Mix channels with a 1x1 conv, apply dropout, and return
        mixed = self.mix_conv(merged)
        out = self.dropout(mixed)
        return out

# Configuration / default sizes
BATCH = 8
SEQ_LEN = 16    # final sequence length after paddings in volumetric branch
SEQ_CHANNELS = 4
VOL_CHANNELS = 3
DEPTH = 14      # initial depth of volumetric input; padding front/back = 1 -> DEPTH + 2 = SEQ_LEN
HEIGHT = 8
WIDTH = 8
MID_CHANNELS = 6
OUT_CHANNELS = 10
DECONV_KERNEL = 3
PAD_3D = (0, 0, 0, 0, 1, 1)  # pad front and back by 1 in depth dimension

def get_inputs():
    """
    Returns inputs matching Model.forward signature:
    - seq: (BATCH, SEQ_CHANNELS, SEQ_LEN)
    - vol: (BATCH, VOL_CHANNELS, DEPTH, HEIGHT, WIDTH)
    """
    seq = torch.randn(BATCH, SEQ_CHANNELS, SEQ_LEN)
    vol = torch.randn(BATCH, VOL_CHANNELS, DEPTH, HEIGHT, WIDTH)
    return [seq, vol]

def get_init_inputs():
    """
    Returns initialization parameters for the Model constructor in the order:
    seq_channels, vol_channels, mid_channels, out_channels, deconv_kernel, pad_3d
    """
    return [SEQ_CHANNELS, VOL_CHANNELS, MID_CHANNELS, OUT_CHANNELS, DECONV_KERNEL, PAD_3D]