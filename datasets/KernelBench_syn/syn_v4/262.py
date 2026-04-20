import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex kernel-like module that:
    - Pads an image tensor using replication padding
    - Splits the padded image into non-overlapping patches (unfold)
    - Projects patches into a transformer embedding space
    - Runs a single TransformerDecoderLayer using a small learned-like memory derived from the sequence
    - Applies a ReLU and projects back to patch pixels
    - Reconstructs the image (fold) and crops to the original size

    The module demonstrates combining ReplicationPad2d, TransformerDecoderLayer, and ReLU in a small pipeline.
    """
    def __init__(self):
        super(Model, self).__init__()
        # Use module-level configuration values
        self.patch_size = PATCH_SIZE
        self.padding = PADDING
        self.in_channels = IN_CHANNELS
        self.d_model = D_MODEL
        self.nhead = NHEAD
        # Linear projections to/from transformer embedding space
        self.patch_dim = self.in_channels * (self.patch_size ** 2)
        self.proj_in = nn.Linear(self.patch_dim, self.d_model)
        self.proj_out = nn.Linear(self.d_model, self.patch_dim)
        # Replication padding
        self.pad = nn.ReplicationPad2d(self.padding)
        # Transformer decoder layer (single layer)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model, nhead=self.nhead)
        # Non-linearity
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input image tensor, shape (B, C, H, W)

        Returns:
            torch.Tensor: Reconstructed image after transformer patch processing, shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        # 1) Pad the input spatially using replication padding
        x_padded = self.pad(x)  # (B, C, H+2p, W+2p)
        H_padded = H + 2 * self.padding
        W_padded = W + 2 * self.padding

        # 2) Extract non-overlapping patches -> Unfold
        # Result: (B, C * patch_h * patch_w, L) where L = num_patches
        patches = F.unfold(x_padded, kernel_size=self.patch_size, stride=self.patch_size)
        # 3) Prepare sequence for transformer: (L, B, patch_dim)
        patches_seq = patches.permute(2, 0, 1)  # (L, B, patch_dim)

        # 4) Project patches into d_model embedding space
        # Linear supports (..., in_features) so we can apply directly
        embedded = self.proj_in(patches_seq)  # (L, B, d_model)

        # 5) Build a small "memory" for cross-attention:
        #    Use the mean across sequence as a compact context token and repeat it.
        memory_token = embedded.mean(dim=0, keepdim=True)  # (1, B, d_model)
        memory_len = MEMORY_LEN  # small number of memory tokens (configurable)
        memory = memory_token.repeat(memory_len, 1, 1)  # (memory_len, B, d_model)

        # 6) Transformer decoder layer: tgt=embedded sequence, memory=context tokens
        decoded = self.decoder_layer(tgt=embedded, memory=memory)  # (L, B, d_model)

        # 7) Non-linearity
        activated = self.relu(decoded)  # (L, B, d_model)

        # 8) Project back to patch pixel space
        back_patches_seq = self.proj_out(activated)  # (L, B, patch_dim)
        # Permute to (B, patch_dim, L) for folding
        back_patches = back_patches_seq.permute(1, 2, 0)  # (B, patch_dim, L)

        # 9) Reconstruct the padded image via fold
        reconstructed_padded = F.fold(back_patches,
                                      output_size=(H_padded, W_padded),
                                      kernel_size=self.patch_size,
                                      stride=self.patch_size)  # (B, C, H_padded, W_padded)

        # 10) Crop to original size (remove padding)
        if self.padding > 0:
            reconstructed = reconstructed_padded[:, :, self.padding:self.padding+H, self.padding:self.padding+W]
        else:
            reconstructed = reconstructed_padded

        return reconstructed

# Configuration variables
BATCH_SIZE = 8
IN_CHANNELS = 3
HEIGHT = 32
WIDTH = 32
PATCH_SIZE = 4
PADDING = 2

# Transformer config (d_model must be divisible by nhead)
D_MODEL = 128
NHEAD = 8
# Memory length for cross-attention in the decoder
MEMORY_LEN = 2

def get_inputs():
    """
    Returns a list with a single input tensor matching the configured shapes:
    (BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)
    """
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)
    return [x]

def get_init_inputs():
    """
    No constructor parameters needed; all configuration is module-level.
    """
    return []