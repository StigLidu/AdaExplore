import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex multi-modal module that fuses a 2D image tensor with a 1D sequence
    by:
      - Upsampling the image with bilinear interpolation (nn.UpsamplingBilinear2d)
      - Reducing the sequence via 1D average pooling (nn.AvgPool1d)
      - Broadcasting the pooled sequence to form a depth dimension and modulating
        the upsampled image to create a 5D tensor
      - Applying 3D average pooling over the fused depth/spatial tensor (nn.AvgPool3d)
      - Global spatial-depth reduction to produce a compact per-channel embedding
    The module demonstrates combining UpsamplingBilinear2d, AvgPool1d and AvgPool3d
    in a non-trivial fusion pattern.
    """
    def __init__(
        self,
        upsample_scale: int,
        pool1d_kernel: int,
        pool3d_kernel: int,
        pool3d_stride: int,
        pool3d_padding: int
    ):
        """
        Initialize the fusion module.

        Args:
            upsample_scale (int): Bilinear upsampling scale factor for height/width.
            pool1d_kernel (int): Kernel size for AvgPool1d applied to the sequence.
            pool3d_kernel (int): Kernel size for AvgPool3d (applied to depth, height, width).
            pool3d_stride (int): Stride for AvgPool3d.
            pool3d_padding (int): Padding for AvgPool3d.
        """
        super(Model, self).__init__()
        # Upsampling layer for 2D images
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=upsample_scale)
        # 1D average pooling to downsample the temporal sequence
        self.pool1d = nn.AvgPool1d(kernel_size=pool1d_kernel, stride=pool1d_kernel)
        # 3D average pooling to operate on the fused (C, D, H, W) tensor
        self.pool3d = nn.AvgPool3d(kernel_size=pool3d_kernel, stride=pool3d_stride, padding=pool3d_padding)
        # Small channel projection after pooling to mix channel information
        # (keeps design compact and demonstrates a common pattern)
        self.channel_proj = nn.Linear(128, 128)  # fixed-size projection applied after flatten reduction
        # Non-linearity
        self.relu = nn.ReLU(inplace=True)

    def forward(self, img: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining image and sequence.

        Args:
            img (torch.Tensor): Image tensor of shape (B, C_img, H, W).
            seq (torch.Tensor): Sequence tensor of shape (B, C_seq, L).

        Returns:
            torch.Tensor: Per-batch per-channel embedding tensor of shape (B, C_img)
                          after fusion and reductions.
        """
        # 1) Upsample the image (B, C_img, H', W')
        up_img = self.upsample(img)

        # 2) Pool the sequence to reduce temporal resolution (B, C_seq, L')
        pooled_seq = self.pool1d(seq)

        # 3) Create a depth axis from pooled sequence:
        #    Compute a summary over sequence channels to get a per-depth scalar for each batch.
        #    shapes: pooled_seq -> (B, C_seq, L') -> seq_summary -> (B, L')
        seq_summary = pooled_seq.mean(dim=1)  # mean over sequence channels

        # 4) Normalize the per-depth summary per-sample to avoid scale issues
        #    shape remains (B, L')
        max_vals = seq_summary.amax(dim=1, keepdim=True)  # (B, 1)
        # Prevent division by zero
        max_vals = max_vals.clamp(min=1e-6)
        seq_norm = seq_summary / max_vals  # (B, L')

        # 5) Broadcast normalized depth weights to (B, 1, D, 1, 1) for multiplication
        depth_weights = seq_norm.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # (B, 1, D, 1, 1)

        # 6) Expand upsampled image to include a depth axis and modulate by depth_weights:
        #    up_img.unsqueeze(2): (B, C_img, 1, H', W')
        #    multiplication broadcasts to (B, C_img, D, H', W')
        fused_5d = up_img.unsqueeze(2) * depth_weights  # fused 5D tensor

        # 7) Apply 3D average pooling to the fused tensor -> reduces (D, H', W')
        pooled_3d = self.pool3d(fused_5d)  # (B, C_img, D_out, H_out, W_out)

        # 8) Global reduction: average across depth and spatial dimensions to get per-channel embeddings
        #    result shape: (B, C_img)
        per_channel = pooled_3d.mean(dim=[2, 3, 4])

        # 9) Optional small projection + activation to mix channels (demonstrates extra op)
        #    To keep projection dimensions consistent, pad or truncate channels to 128.
        B, C = per_channel.shape
        if C < 128:
            # pad with zeros to 128
            pad = per_channel.new_zeros(B, 128 - C)
            proj_input = torch.cat([per_channel, pad], dim=1)
        else:
            proj_input = per_channel[:, :128]

        projected = self.channel_proj(proj_input)
        activated = self.relu(projected)

        # 10) Return the activated projection trimmed back to original channel count
        if C < 128:
            return activated[:, :C]
        else:
            return activated[:, :C]

# Configuration variables
batch_size = 8
img_channels = 3
height = 64
width = 64

seq_channels = 8
seq_length = 32

upsample_scale = 2            # bilinear upsampling factor
pool1d_kernel = 4             # reduces seq_length 32 -> 8
pool3d_kernel = 2
pool3d_stride = 2
pool3d_padding = 0

def get_inputs():
    """
    Returns example inputs matching the module's expected input signatures:
      - img: (batch_size, img_channels, height, width)
      - seq: (batch_size, seq_channels, seq_length)
    """
    img = torch.randn(batch_size, img_channels, height, width)
    seq = torch.randn(batch_size, seq_channels, seq_length)
    return [img, seq]

def get_init_inputs():
    """
    Returns the initialization parameters used to construct the Model.
    """
    return [upsample_scale, pool1d_kernel, pool3d_kernel, pool3d_stride, pool3d_padding]