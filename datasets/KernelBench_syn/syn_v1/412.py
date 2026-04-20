import torch
import torch.nn as nn

# Module-level configuration
BATCH_SIZE = 8
CHANNELS = 64
VIDEO_DEPTH = 8
VIDEO_HEIGHT = 32
VIDEO_WIDTH = 32
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32

# Initialization configuration defaults
POOL_OUTPUT_SIZE = (2, 4, 4)   # (D_out, H_out, W_out) for AdaptiveAvgPool3d
PAD = 0                        # replication padding for ReplicationPad2d (int or tuple)
PATCH_SIZE = 4                 # patch size for unfolding image into non-overlapping patches
HIDDEN_DIM = 256               # hidden dimension for patch projection


class Model(nn.Module):
    """
    Complex module that combines 3D adaptive average pooling on a video tensor
    with 2D replication padding + patch-based processing on an image tensor,
    then fuses both streams into a consolidated per-sample feature vector.

    Pipeline:
      - Video branch:
          AdaptiveAvgPool3d -> average over temporal dimension -> spatial global average -> per-channel vector
      - Image branch:
          ReplicationPad2d -> Unfold into patches -> linear projection of patches -> aggregate patch features
      - Fusion:
          Project pooled video vector to hidden_dim, elementwise multiply with aggregated patch features,
          final linear layer projects back to per-channel output.
    """
    def __init__(
        self,
        in_channels: int,
        pool_output_size: tuple,
        pad: int,
        patch_size: int,
        hidden_dim: int,
    ):
        """
        Args:
            in_channels (int): Number of input channels for both video and image tensors.
            pool_output_size (tuple): Output size for AdaptiveAvgPool3d (D_out, H_out, W_out).
            pad (int or tuple): Padding to use for ReplicationPad2d.
            patch_size (int): Size of square patches for unfolding the image.
            hidden_dim (int): Hidden dimensionality for patch projection and fusion.
        """
        super(Model, self).__init__()
        # Video branch pooling
        self.pool3d = nn.AdaptiveAvgPool3d(pool_output_size)

        # Image branch padding and unfolding
        self.pad2d = nn.ReplicationPad2d(pad)
        self.patch_size = patch_size
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)

        # Linear layers for projection
        # patch vector dimension = in_channels * patch_size * patch_size
        patch_dim = in_channels * patch_size * patch_size
        self.fc_patch = nn.Linear(patch_dim, hidden_dim)     # projects each patch to hidden_dim
        self.pool_proj = nn.Linear(in_channels, hidden_dim)  # projects per-channel pooled vector to hidden_dim

        # Fusion and final projection
        self.fc_fusion_to_channels = nn.Linear(hidden_dim, in_channels)

        # small activation and normalization to stabilize
        self.activation = nn.GELU()
        self.layernorm = nn.LayerNorm(hidden_dim)

    def forward(self, video: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video (torch.Tensor): Video tensor of shape (N, C, D, H, W).
            image (torch.Tensor): Image tensor of shape (N, C, H_img, W_img).

        Returns:
            torch.Tensor: Output tensor of shape (N, C) representing per-sample per-channel fused features.
        """
        N, C, D, H, W = video.shape

        # 1) Video branch: Adaptive 3D average pooling
        pooled_video = self.pool3d(video)                       # (N, C, Dp, Hp, Wp)
        # 2) Reduce temporal dimension by averaging across Dp
        pooled_video_spatial = pooled_video.mean(dim=2)         # (N, C, Hp, Wp)
        # 3) Global spatial average to get per-channel vector
        pooled_video_vector = pooled_video_spatial.view(N, C, -1).mean(dim=2)  # (N, C)

        # 4) Image branch: replication padding then extract non-overlapping patches
        padded_image = self.pad2d(image)                        # (N, C, H_pad, W_pad)
        patches = self.unfold(padded_image)                     # (N, C * p*p, L) where L is number of patches
        # 5) Transpose and project each patch
        patches_t = patches.transpose(1, 2)                     # (N, L, patch_dim)
        patch_feats = self.fc_patch(patches_t)                  # (N, L, hidden_dim)
        patch_feats = self.activation(patch_feats)
        # 6) Aggregate patch features spatially (mean over patches) and normalize
        agg_patch = patch_feats.mean(dim=1)                     # (N, hidden_dim)
        agg_patch = self.layernorm(agg_patch)

        # 7) Project pooled video vector to hidden_dim
        proj_video = self.pool_proj(pooled_video_vector)        # (N, hidden_dim)
        proj_video = self.activation(proj_video)

        # 8) Elementwise fusion
        fused = agg_patch * proj_video                          # (N, hidden_dim)
        fused = self.activation(fused)

        # 9) Final projection back to per-channel outputs
        out = self.fc_fusion_to_channels(fused)                 # (N, C)

        return out


def get_inputs():
    """
    Returns:
        list: [video_tensor, image_tensor]
            - video_tensor: (BATCH_SIZE, CHANNELS, VIDEO_DEPTH, VIDEO_HEIGHT, VIDEO_WIDTH)
            - image_tensor: (BATCH_SIZE, CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
    """
    video = torch.randn(BATCH_SIZE, CHANNELS, VIDEO_DEPTH, VIDEO_HEIGHT, VIDEO_WIDTH)
    image = torch.randn(BATCH_SIZE, CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
    return [video, image]


def get_init_inputs():
    """
    Returns initialization parameters for Model:
      [in_channels, pool_output_size, pad, patch_size, hidden_dim]
    """
    return [CHANNELS, POOL_OUTPUT_SIZE, PAD, PATCH_SIZE, HIDDEN_DIM]