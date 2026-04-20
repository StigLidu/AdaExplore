import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A 1D patch-based transformer-lite block that:
    - Zero-pads the input temporally
    - Extracts overlapping 1D patches via nn.Unfold (treating width=1)
    - Applies a two-layer MLP with GELU nonlinearity per-patch
    - Reconstructs the sequence with nn.Fold and normalizes overlapping contributions

    This demonstrates a combination of ZeroPad1d, nn.Unfold/nn.Fold and nn.GELU
    to create a non-trivial patch extraction and reconstruction pattern.
    """
    def __init__(
        self,
        in_channels: int = 3,
        length: int = 128,
        kernel_size: int = 5,
        stride: int = 2,
        padding: int = 2,
        hidden_dim: int = 64,
        out_channels: int = 4,
    ):
        """
        Initializes the patch-based model.

        Args:
            in_channels (int): Number of input channels.
            length (int): Original 1D length of the sequence.
            kernel_size (int): Size of each 1D patch.
            stride (int): Stride between patches.
            padding (int): Zero-padding applied to both sides of the sequence.
            hidden_dim (int): Hidden dimension of the per-patch MLP.
            out_channels (int): Number of output channels after reconstruction.
        """
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.length = length
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels

        # Pad along the temporal (last) dimension
        # ZeroPad1d pads (left, right) on the last dimension of (N, C, L)
        self.pad = nn.ZeroPad1d(self.padding)

        # Treat the 1D signal as a (H, W) image with W=1 to reuse Unfold/Fold (2D).
        # Extract patches of shape (kernel_size, 1) with given stride.
        self.unfold = nn.Unfold(kernel_size=(self.kernel_size, 1), stride=(self.stride, 1))

        # Fold will reconstruct to the padded length (length + 2*padding) along H, W=1.
        padded_length = self.length + 2 * self.padding
        self.fold = nn.Fold(output_size=(padded_length, 1), kernel_size=(self.kernel_size, 1), stride=(self.stride, 1))

        # Per-patch MLP: map (in_channels * kernel_size) -> hidden_dim -> (out_channels * kernel_size)
        self.fc1 = nn.Linear(self.in_channels * self.kernel_size, self.hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(self.hidden_dim, self.out_channels * self.kernel_size)

        # Small initialization for stability
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C_in, L)

        Returns:
            torch.Tensor: Output tensor of shape (N, C_out, L) (original length)
        """
        # x: (N, C_in, L)
        N = x.shape[0]

        # 1) Zero-pad temporally: (N, C_in, L_p)
        x_padded = self.pad(x)

        # 2) Make it 4D to use Unfold: (N, C, H, W) with W=1
        x_padded_4d = x_padded.unsqueeze(-1)  # (N, C_in, L_p, 1)

        # 3) Extract overlapping patches: (N, C_in * kernel_size, L_out)
        patches = self.unfold(x_padded_4d)  # (N, C_in * k, L_out)

        # 4) Rearrange to process each patch as a feature vector: (N * L_out, C_in * k)
        patches_t = patches.transpose(1, 2)  # (N, L_out, C_in * k)
        N, L_out, feat_dim = patches_t.shape
        patches_flat = patches_t.reshape(N * L_out, feat_dim)  # (N*L_out, feat_dim)

        # 5) Per-patch MLP with GELU: (N*L_out, hidden) -> (N*L_out, out_channels * k)
        hidden = self.fc1(patches_flat)
        hidden = self.gelu(hidden)
        out_patches_flat = self.fc2(hidden)  # (N*L_out, out_channels * k)

        # 6) Reshape back to Fold input shape: (N, out_channels * k, L_out)
        out_patches = out_patches_flat.reshape(N, L_out, self.out_channels * self.kernel_size).transpose(1, 2)

        # 7) Reconstruct the padded sequence via Fold: (N, out_channels, L_p, 1)
        reconstructed = self.fold(out_patches)  # (N, out_channels, L_p, 1)

        # 8) Compute overlap-counts to normalize overlapping sums:
        # Create ones for a single channel to count how many times each position was written
        ones = torch.ones((N, 1 * self.kernel_size, L_out), dtype=reconstructed.dtype, device=reconstructed.device)
        denom = self.fold(ones)  # (N, 1, L_p, 1)
        denom = denom.clamp(min=1.0)

        # Normalize by overlap counts (broadcast over channels)
        normalized = reconstructed / denom

        # 9) Remove the singleton width dimension and unpad back to original length
        normalized = normalized.squeeze(-1)  # (N, out_channels, L_p)
        start = self.padding
        end = start + self.length
        out = normalized[:, :, start:end]  # (N, out_channels, L)

        return out

# Configuration (module-level)
batch_size = 8
in_channels = 3
length = 128
kernel_size = 5
stride = 2
padding = 2
hidden_dim = 64
out_channels = 4

def get_inputs():
    """
    Returns a list containing a single input tensor of shape (batch_size, in_channels, length).
    """
    x = torch.randn(batch_size, in_channels, length)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for the Model constructor in the same order:
    (in_channels, length, kernel_size, stride, padding, hidden_dim, out_channels)
    """
    return [in_channels, length, kernel_size, stride, padding, hidden_dim, out_channels]