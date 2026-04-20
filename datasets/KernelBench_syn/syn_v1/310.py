import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex model combining 1D replication padding + Conv1d + Tanhshrink branch
    and a 3D convolutional branch that uses MaxPool3d with indices and MaxUnpool3d
    to reconstruct spatial structure. The two branches produce feature vectors
    that are concatenated and passed through a final fully connected layer.

    Forward signature:
        forward(seq: Tensor, vol: Tensor) -> Tensor

    Inputs:
        seq: Tensor of shape (batch_size, seq_channels, seq_length)
        vol: Tensor of shape (batch_size, vol_channels, D, H, W)
    """
    def __init__(
        self,
        seq_channels: int,
        seq_pad: int,
        conv1_out: int,
        conv1_kernel: int,
        vol_channels: int,
        pool_k: int,
        pool_stride: int,
        final_dim: int = 256
    ):
        super(Model, self).__init__()

        # 1D branch: replication padding -> conv1d -> tanhshrink -> adaptive avg pool -> fc
        self.seq_pad = seq_pad
        self.pad1d = nn.ReplicationPad1d(seq_pad)
        self.conv1d = nn.Conv1d(in_channels=seq_channels, out_channels=conv1_out, kernel_size=conv1_kernel, stride=1)
        self.act = nn.Tanhshrink()
        self.avgpool1d = nn.AdaptiveAvgPool1d(1)
        self.fc_seq = nn.Linear(conv1_out, 64)

        # 3D branch: conv3d -> maxpool3d(return_indices=True) -> maxunpool3d -> adaptive avg pool -> fc
        self.conv3d = nn.Conv3d(in_channels=vol_channels, out_channels=16, kernel_size=3, padding=1)
        self.maxpool3d = nn.MaxPool3d(kernel_size=pool_k, stride=pool_stride, return_indices=True)
        self.unpool3d = nn.MaxUnpool3d(kernel_size=pool_k, stride=pool_stride)
        self.avgpool3d = nn.AdaptiveAvgPool3d(1)
        self.fc_vol = nn.Linear(16, 64)

        # Fusion and final projection
        self.fc_fusion = nn.Linear(64 + 64, final_dim)

    def forward(self, seq: torch.Tensor, vol: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            seq: (B, seq_channels, L)
            vol: (B, vol_channels, D, H, W)

        Returns:
            Tensor: (B, final_dim)
        """
        # 1D branch
        # Replication padding (pads both sides by seq_pad)
        x = self.pad1d(seq)
        x = self.conv1d(x)
        x = self.act(x)
        # Adaptive pool to get a single vector per channel
        x = self.avgpool1d(x)  # (B, conv1_out, 1)
        x = x.view(x.size(0), -1)  # (B, conv1_out)
        x = self.fc_seq(x)  # (B, 64)

        # 3D branch
        y = self.conv3d(vol)  # (B, 16, D, H, W)
        # Save size for unpooling
        y_size = y.size()
        y_pooled, indices = self.maxpool3d(y)  # pooled size smaller
        # Unpool back to original conv3d output size using indices
        y_unpooled = self.unpool3d(y_pooled, indices, output_size=y_size)
        # Adaptive avg pool to (1,1,1) and flatten
        y_vec = self.avgpool3d(y_unpooled).view(y_unpooled.size(0), -1)  # (B, 16)
        y = self.fc_vol(y_vec)  # (B, 64)

        # Fuse and final projection
        fused = torch.cat([x, y], dim=1)  # (B, 128)
        out = self.fc_fusion(fused)  # (B, final_dim)
        return out

# Configuration (module-level) variables
BATCH_SIZE = 8
SEQ_CHANNELS = 16
SEQ_LEN = 128
SEQ_PAD = 2
CONV1_OUT = 32
CONV1_KERNEL = 5

VOL_CHANNELS = 8
D = 16
H = 32
W = 32
POOL_K = 2
POOL_STRIDE = 2

FINAL_DIM = 256

def get_inputs():
    """
    Returns runtime input tensors for the model:
        - seq: (BATCH_SIZE, SEQ_CHANNELS, SEQ_LEN)
        - vol: (BATCH_SIZE, VOL_CHANNELS, D, H, W)
    """
    seq = torch.randn(BATCH_SIZE, SEQ_CHANNELS, SEQ_LEN)
    vol = torch.randn(BATCH_SIZE, VOL_CHANNELS, D, H, W)
    return [seq, vol]

def get_init_inputs():
    """
    Returns initialization parameters matching the Model.__init__ signature:
        (seq_channels, seq_pad, conv1_out, conv1_kernel, vol_channels, pool_k, pool_stride, final_dim)
    """
    return [SEQ_CHANNELS, SEQ_PAD, CONV1_OUT, CONV1_KERNEL, VOL_CHANNELS, POOL_K, POOL_STRIDE, FINAL_DIM]