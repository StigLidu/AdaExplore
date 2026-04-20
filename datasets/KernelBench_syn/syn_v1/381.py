import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class Model(nn.Module):
    """
    Complex vision-to-classification model that:
    - Upsamples an input image with bilinear interpolation
    - Extracts non-overlapping patches and linearly projects them to embeddings
    - Runs a MultiheadAttention over the patch sequence
    - Pools the attended sequence and computes an AdaptiveLogSoftmaxWithLoss

    The model returns the scalar loss computed by AdaptiveLogSoftmaxWithLoss when
    provided with targets.
    """
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        num_heads: int,
        n_classes: int,
        cutoffs: List[int],
        patch_size: int = 4,
        upsample_scale: int = 2,
    ):
        """
        Args:
            in_channels: Number of input image channels.
            embed_dim: Embedding dimension for patches and attention.
            num_heads: Number of attention heads.
            n_classes: Total number of classes for adaptive softmax.
            cutoffs: Cutoff boundaries for AdaptiveLogSoftmaxWithLoss.
            patch_size: Size of non-overlapping patches (patch_size x patch_size).
            upsample_scale: Bilinear upsampling scale factor.
        """
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=upsample_scale)
        
        # Linear projection from flattened patch to embedding dim
        proj_in_dim = in_channels * patch_size * patch_size
        self.embed_proj = nn.Linear(proj_in_dim, embed_dim)
        
        # Multihead attention over patch sequence
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        
        # Small feedforward projection after attention (residual-aware)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Adaptive softmax for large-vocab classification with efficient loss
        self.adaptive_log_softmax = nn.AdaptiveLogSoftmaxWithLoss(in_features=embed_dim,
                                                                 n_classes=n_classes,
                                                                 cutoffs=cutoffs)
    
    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing the adaptive softmax loss.

        Args:
            x: Input image tensor of shape (B, C, H, W).
            target: Target class indices of shape (B,).

        Returns:
            torch.Tensor: Scalar loss tensor from AdaptiveLogSoftmaxWithLoss.
        """
        # 1) Upsample spatially
        x_up = self.upsample(x)  # (B, C, H*scale, W*scale)

        # 2) Extract non-overlapping patches and flatten per patch
        #    F.unfold -> (B, C*patch*patch, L) where L = num_patches
        patches = F.unfold(x_up, kernel_size=self.patch_size, stride=self.patch_size)
        # (B, L, patch_flat_dim)
        patches = patches.transpose(1, 2)

        # 3) Linear projection to embeddings: (B, L, E)
        emb = self.embed_proj(patches)

        # 4) Prepare for MultiheadAttention which expects (L, B, E)
        seq = emb.transpose(0, 1)

        # 5) Self-attention over patch sequence
        attn_out, _ = self.attn(seq, seq, seq)  # (L, B, E)

        # 6) Residual + Feedforward + LayerNorm (classic transformer block)
        seq_res = self.layer_norm(attn_out + seq)  # (L, B, E)
        ff_out = self.ff(seq_res)  # (L, B, E)
        seq_out = self.layer_norm(seq_res + ff_out)  # (L, B, E)

        # 7) Pool across sequence (mean pooling) -> (B, E)
        pooled = seq_out.mean(dim=0)

        # 8) Compute adaptive log-softmax loss
        loss_output = self.adaptive_log_softmax(pooled, target)
        # loss_output is a namedtuple with .output (log-probs) and .loss (scalar)
        return loss_output.loss

# Configuration / default inputs
batch_size = 8
in_channels = 3
height = 32
width = 32

embed_dim = 128
num_heads = 8
n_classes = 10000
cutoffs = [2000, 6000]  # example cutoffs for adaptive softmax
patch_size = 4
upsample_scale = 2

def get_inputs():
    """
    Returns a list containing:
      - input image tensor x of shape (batch_size, in_channels, height, width)
      - target tensor of shape (batch_size,) with class indices in [0, n_classes)
    """
    x = torch.randn(batch_size, in_channels, height, width)
    target = torch.randint(low=0, high=n_classes, size=(batch_size,))
    return [x, target]

def get_init_inputs():
    """
    Returns initialization arguments for Model in the order:
      in_channels, embed_dim, num_heads, n_classes, cutoffs, patch_size, upsample_scale
    """
    return [in_channels, embed_dim, num_heads, n_classes, cutoffs, patch_size, upsample_scale]