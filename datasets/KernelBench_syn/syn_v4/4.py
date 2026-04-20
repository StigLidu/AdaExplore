import torch
import torch.nn as nn
import torch.nn.functional as F

# Configuration (module-level)
BATCH_SIZE = 8
IN_CHANNELS = 3
HEIGHT = 64
WIDTH = 48
PATCH_SIZE = 4        # spatial patch size (must divide padded H and W)
PROJ_DIM = 128        # per-patch projection dimension
OUT_CHANNELS = 6      # output channels after reconstruction
PAD = 0               # constant pad amount (applied symmetrically)

class Model(nn.Module):
    """
    Patch-based extractor-transformer-fuser:
      - Pads the input spatially using ConstantPad2d
      - Extracts non-overlapping patches with unfold (kernel = PATCH_SIZE)
      - Applies RMSNorm over the flattened patch vector
      - Projects each patch to a feature vector (linear)
      - Smooths features across neighboring patches with ReplicationPad1d + avg_pool1d
      - Computes an attention-weighted global descriptor from smoothed patch features
      - Concatenates local and global features, projects back to patch pixels
      - Reconstructs an output image with OUT_CHANNELS using fold

    Inputs:
      x: Tensor of shape (B, IN_CHANNELS, H, W)

    Output:
      Tensor of shape (B, OUT_CHANNELS, H_out, W_out) where H_out/W_out reflect padding.
    """
    def __init__(self, in_channels: int, patch_size: int, proj_dim: int, out_channels: int, pad: int = 0):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.proj_dim = proj_dim
        self.out_channels = out_channels
        self.pad = pad

        # Padding layer (constant)
        # ConstantPad2d takes pad as (left, right, top, bottom) or int
        self.pad2d = nn.ConstantPad2d(pad, 0.0)

        # After unfolding, each patch is a vector of this dimensionality:
        self.patch_dim = in_channels * (patch_size * patch_size)

        # RMS normalization over the flattened patch vector
        self.rmsnorm = nn.RMSNorm(self.patch_dim, eps=1e-6)

        # Project normalized patch -> proj_dim
        self.linear1 = nn.Linear(self.patch_dim, proj_dim, bias=True)

        # Replication pad for smoothing across the patch-sequence (1D)
        # We'll pad 1 on both sides (kernel smoothing of neighbouring patches)
        self.repl_pad1d = nn.ReplicationPad1d((1, 1))

        # Final projection: concatenate local proj and global descriptor -> reconstruct patch pixels for OUT_CHANNELS
        # Output per patch size = out_channels * patch_size * patch_size
        self.reconstruct_dim = out_channels * (patch_size * patch_size)
        self.linear2 = nn.Linear(proj_dim * 2, self.reconstruct_dim, bias=True)

        # Learnable query vector for attention scoring
        self.query = nn.Parameter(torch.randn(proj_dim))

        # Initialize weights in a simple, controlled manner
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)
        nn.init.normal_(self.query, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            out: Reconstructed tensor of shape (B, out_channels, H_padded, W_padded)
        """
        # 1) Constant pad spatially
        x_p = self.pad2d(x)  # (B, C, H_padded, W_padded)
        B, C, H_p, W_p = x_p.shape

        # 2) Extract non-overlapping patches as flattened vectors
        #    patches: (B, patch_dim, L) where L = number of patches
        patches = F.unfold(x_p, kernel_size=self.patch_size, stride=self.patch_size)  # (B, D, L)
        patches = patches.transpose(1, 2)  # (B, L, D) where D = patch_dim

        # 3) RMS normalization over the flattened patch vector dimension
        patches_norm = self.rmsnorm(patches)  # (B, L, D)

        # 4) Project each patch to a feature vector
        proj = self.linear1(patches_norm)  # (B, L, proj_dim)

        # 5) Smooth features across neighboring patches using ReplicationPad1d + average pooling
        #    Convert to (B, proj_dim, L) for 1D padding/pooling
        proj_perm = proj.permute(0, 2, 1)  # (B, proj_dim, L)
        proj_padded = self.repl_pad1d(proj_perm)  # (B, proj_dim, L+2)
        # Average over windows of size 3 to smooth neighbors, stride=1 to keep length L
        proj_smooth_perm = F.avg_pool1d(proj_padded, kernel_size=3, stride=1)  # (B, proj_dim, L)
        proj_smooth = proj_smooth_perm.permute(0, 2, 1)  # (B, L, proj_dim)

        # 6) Attention-like global descriptor over patches
        #    scores: (B, L)
        #    Use a dot product with learnable query
        scores = torch.matmul(proj_smooth, self.query) / (self.proj_dim ** 0.5)
        attn = F.softmax(scores, dim=1)  # (B, L)
        # Weighted sum to get global descriptor per batch: (B, proj_dim)
        global_desc = torch.sum(proj_smooth * attn.unsqueeze(-1), dim=1)

        # 7) Concatenate local and global features and reconstruct patch pixels
        global_expanded = global_desc.unsqueeze(1).expand(-1, proj_smooth.shape[1], -1)  # (B, L, proj_dim)
        concat = torch.cat([proj_smooth, global_expanded], dim=-1)  # (B, L, 2*proj_dim)

        out_patches = self.linear2(concat)  # (B, L, reconstruct_dim)
        # Prepare for folding: (B, reconstruct_dim, L)
        out_patches = out_patches.transpose(1, 2)  # (B, reconstruct_dim, L)

        # 8) Fold back to spatial map to get (B, out_channels, H_p, W_p)
        out = F.fold(out_patches,
                     output_size=(H_p, W_p),
                     kernel_size=self.patch_size,
                     stride=self.patch_size)

        # The fold produces a tensor shaped (B, out_channels * 1, H_p, W_p) given how reconstruct_dim was structured.
        # Because reconstruct_dim = out_channels * patch_size * patch_size and fold sums patches into channel groups,
        # the output will be (B, out_channels, H_p, W_p). If additional scaling is needed, it can be applied here.
        return out


# Test configuration values
batch_size = BATCH_SIZE
in_channels = IN_CHANNELS
height = HEIGHT
width = WIDTH
patch_size = PATCH_SIZE
proj_dim = PROJ_DIM
out_channels = OUT_CHANNELS
pad = PAD

def get_inputs():
    # Create a random input tensor matching the prescribed sizes
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    # Return initialization parameters for the Model constructor
    return [in_channels, patch_size, proj_dim, out_channels, pad]