import torch
import torch.nn as nn
import torch.nn.functional as F

# Configuration
BATCH_SIZE = 8
IN_CHANNELS = 3
HEIGHT = 32
WIDTH = 32
KERNEL = 3  # kernel size for Unfold and internal processing

class Model(nn.Module):
    """
    A composite module that:
      - Extracts sliding patches from an input image using nn.Unfold
      - Applies a small patch-wise normalization/gating
      - Processes the sequence of patches with two sequential ConvTranspose1d layers
        to progressively increase the spatial resolution (sequence length)
      - Uses HardTanh as a non-linearity between the ConvTranspose1d layers
      - Reshapes the final sequence back into an image-like tensor

    Input:
      x: Tensor of shape (BATCH, IN_CHANNELS, HEIGHT, WIDTH)

    Output:
      Tensor of shape (BATCH, OUT_CHANNELS, HEIGHT, WIDTH) with OUT_CHANNELS = conv2_out_channels
    """
    def __init__(self):
        super(Model, self).__init__()
        # Unfold parameters: kernel_size=KERNEL, stride=2, padding=1 -> reduces spatial dims roughly by 2
        self.unfold = nn.Unfold(kernel_size=KERNEL, stride=2, padding=1)
        # patch feature dimension
        self.patch_dim = IN_CHANNELS * (KERNEL * KERNEL)  # e.g., 3 * 3 * 3 = 27

        # First transposed conv: increases sequence length by factor 2 (stride=2)
        # Choose out channels reasonably larger to mix patch features
        self.convT1 = nn.ConvTranspose1d(
            in_channels=self.patch_dim,
            out_channels=128,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )

        # Non-linearity
        self.act = nn.Hardtanh(min_val=-1.0, max_val=1.0)

        # Second transposed conv: increase sequence length again by factor 2 (total factor 4)
        self.convT2 = nn.ConvTranspose1d(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )

        # A small learnable scale to modulate final output intensity
        self.register_parameter("out_scale", nn.Parameter(torch.tensor(0.5)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
          1. Extract patches -> (B, patch_dim, L)
          2. Compute patch norms and create a gating coefficient per patch
          3. Apply gating to patches
          4. Pass through two ConvTranspose1d layers with HardTanh between them
          5. Reshape the final sequence into (B, out_channels, HEIGHT, WIDTH)
        """
        B = x.shape[0]
        # 1) Extract patches
        patches = self.unfold(x)  # shape: (B, patch_dim, L)
        # 2) Patch-wise L2 norm -> (B, 1, L)
        patch_norms = patches.norm(p=2, dim=1, keepdim=True)  # avoid division by zero
        # Normalize norms across the sequence to get gating factors in a stable range
        norm_mean = patch_norms.mean(dim=2, keepdim=True)
        gating = patch_norms / (norm_mean + 1e-6)  # (B,1,L)
        # Clamp gating to avoid extreme scaling
        gating = gating.clamp(0.1, 10.0)
        # 3) Apply gating
        patches_mod = patches * gating  # (B, patch_dim, L)

        # 4) First ConvTranspose1d: increases sequence length
        t1 = self.convT1(patches_mod)  # (B, 128, L1) where L1 ~ 2*L
        a1 = self.act(t1)

        # 5) Second ConvTranspose1d: further increase sequence length
        t2 = self.convT2(a1)  # (B, 64, L2) where L2 ~ 4*L
        a2 = self.act(t2)

        # Now compute expected spatial sizes:
        # After Unfold with stride=2, per-dim positions = ceil(HEIGHT/2), ceil(WIDTH/2)
        Hp = (HEIGHT + 2*1 - KERNEL) // 2 + 1  # floor formula equivalent for these integers
        Wp = (WIDTH + 2*1 - KERNEL) // 2 + 1
        L = Hp * Wp  # original sequence length after unfold
        L2_expected = L * 4  # after two stride=2 upsamples

        # Ensure the width dimension matches HEIGHT*WIDTH; if it doesn't, adapt by cropping/padding
        # Current seq_len:
        seq_len = a2.shape[2]
        target_len = HEIGHT * WIDTH

        if seq_len < target_len:
            # pad on the right
            pad_amount = target_len - seq_len
            a2 = F.pad(a2, (0, pad_amount))
            seq_len = target_len
        elif seq_len > target_len:
            # crop to target_len
            a2 = a2[:, :, :target_len]
            seq_len = target_len

        # 6) Reshape to image: (B, C_out, HEIGHT, WIDTH)
        out_channels = a2.shape[1]  # 64
        out = a2.view(B, out_channels, HEIGHT, WIDTH)

        # 7) Final modulation and non-linearity
        out = self.act(out * self.out_scale)

        return out


def get_inputs():
    """
    Returns a list of input tensors to be passed into Model.forward.
    """
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)
    return [x]

def get_init_inputs():
    """
    Returns initialization inputs if the Model required any non-default parameters.
    For this module we don't require external init parameters.
    """
    return []