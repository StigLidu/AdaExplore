import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex model that combines 1D convolutional feature extraction with 3D instance
    normalization and cross-modal attention-style modulation.

    Architecture summary:
    - 1D branch (sequence input):
        LazyConv1d -> LazyBatchNorm1d -> ReLU -> AdaptiveAvgPool1d -> channel summary
        -> Linear projection -> sigmoid gating vector
    - 3D branch (volume input):
        LazyInstanceNorm3d -> gated modulation by the 1D-produced gating vector
        -> global spatial average
    - Output:
        Concatenation of sequence summary and modulated-volume summary per batch sample.

    The model uses lazy initializers for the in_channels / num_features of conv/bn/instance-norm
    so the channel sizes of inputs are determined at first forward pass.
    """
    def __init__(
        self,
        conv_out_channels: int,
        conv_kernel_size: int,
        seq_pool_size: int,
        vol_channels: int,
        inst_eps: float = 1e-5,
    ):
        """
        Args:
            conv_out_channels: Number of output channels for the 1D convolution.
            conv_kernel_size: Kernel size for the 1D convolution.
            seq_pool_size: Output size for AdaptiveAvgPool1d on the sequence branch.
            vol_channels: Expected number of channels in the 3D volume input; used to size
                          the linear projection that maps sequence features to volume channels.
            inst_eps: Epsilon value for LazyInstanceNorm3d.
        """
        super(Model, self).__init__()
        # 1D feature extractor (lazy in_channels)
        self.conv1d = nn.LazyConv1d(out_channels=conv_out_channels, kernel_size=conv_kernel_size, stride=1, padding=conv_kernel_size // 2)
        self.bn1d = nn.LazyBatchNorm1d()
        self.relu = nn.ReLU(inplace=True)
        # pool the temporal dimension to a fixed small size, then summarize
        self.seq_pool = nn.AdaptiveAvgPool1d(seq_pool_size)
        # projection from conv_out_channels -> vol_channels to create gating for 3D branch
        self.proj_to_vol = nn.Linear(conv_out_channels, vol_channels)
        # 3D instance normalization (lazy num_features)
        self.inst_norm3d = nn.LazyInstanceNorm3d(eps=inst_eps, affine=True)
        # store sizes for reshaping/operations
        self.conv_out_channels = conv_out_channels
        self.seq_pool_size = seq_pool_size
        self.vol_channels = vol_channels

    def forward(self, seq: torch.Tensor, vol: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            seq: 1D sequence tensor of shape (B, C_seq, L)
            vol: 3D volume tensor of shape (B, C_vol, D, H, W)

        Returns:
            Tensor of shape (B, conv_out_channels + vol_channels) containing concatenated
            sequence summary and modulated-volume summary per batch sample.
        """
        # Sequence branch: Conv1d -> BN -> ReLU -> AdaptiveAvgPool1d
        x_seq = self.conv1d(seq)                 # (B, conv_out, L)
        x_seq = self.bn1d(x_seq)                 # (B, conv_out, L)
        x_seq = self.relu(x_seq)                 # (B, conv_out, L)
        x_seq = self.seq_pool(x_seq)             # (B, conv_out, pool_size)

        # Summarize sequence branch: mean across pooled temporal dimension -> (B, conv_out)
        seq_summary = x_seq.mean(dim=2)          # (B, conv_out)

        # Project sequence summary to create gating for volume channels and squash to [0,1]
        gating = self.proj_to_vol(seq_summary)   # (B, vol_channels)
        gating = torch.sigmoid(gating)           # (B, vol_channels)
        # reshape gating to broadcast over (D,H,W)
        gating_map = gating.view(gating.size(0), gating.size(1), 1, 1, 1)  # (B, vol_channels,1,1,1)

        # Volume branch: InstanceNorm3d then gated modulation
        vol_norm = self.inst_norm3d(vol)         # (B, C_vol, D, H, W) normalized per-sample
        # If vol has different channel count than gating, we'll rely on broadcasting rules /
        # expectation that proj_to_vol was sized to vol's channels (vol_channels). If not,
        # gating_map will broadcast or mismatch and raise — this enforces intended usage.
        vol_modulated = vol_norm * gating_map    # (B, C_vol, D, H, W)

        # Global spatial pooling of the modulated volume -> (B, C_vol)
        vol_summary = vol_modulated.mean(dim=(2, 3, 4))  # (B, C_vol)

        # Final output: concatenate sequence summary and volume summary
        out = torch.cat([seq_summary, vol_summary], dim=1)  # (B, conv_out + vol_channels)
        return out


# Configuration / default sizes for test generation
BATCH_SIZE = 8
SEQ_CHANNELS = 12
SEQ_LENGTH = 256

VOL_CHANNELS = 16
DEPTH = 16
HEIGHT = 32
WIDTH = 32

CONV_OUT_CHANNELS = 32
CONV_KERNEL_SIZE = 5
SEQ_POOL_SIZE = 8
INST_EPS = 1e-5

def get_inputs():
    """
    Generate sample inputs for the model:
    - seq: (BATCH_SIZE, SEQ_CHANNELS, SEQ_LENGTH)
    - vol: (BATCH_SIZE, VOL_CHANNELS, DEPTH, HEIGHT, WIDTH)
    """
    seq = torch.randn(BATCH_SIZE, SEQ_CHANNELS, SEQ_LENGTH)
    vol = torch.randn(BATCH_SIZE, VOL_CHANNELS, DEPTH, HEIGHT, WIDTH)
    return [seq, vol]

def get_init_inputs():
    """
    Return initialization arguments to create the Model instance in the order:
    [conv_out_channels, conv_kernel_size, seq_pool_size, vol_channels, inst_eps]
    """
    return [CONV_OUT_CHANNELS, CONV_KERNEL_SIZE, SEQ_POOL_SIZE, VOL_CHANNELS, INST_EPS]