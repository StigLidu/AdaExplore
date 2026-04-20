import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex patch-based feature processor that:
      - Pads the input with replication padding
      - Extracts sliding patches (like im2col / unfold)
      - Projects patches into a hidden space with two separate learnable transforms
      - Applies different nonlinearities (Softplus and Softsign) to each branch
      - Fuses the two branch outputs via a gating-like interaction
      - Projects fused features back to patch space and folds them to reconstruct a spatial tensor
      - Crops the padding to restore the original spatial resolution

    This creates a non-trivial computation graph using ReplicationPad2d, Softplus, Softsign,
    and various tensor reshaping/einsum operations.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int, padding: int, hidden_dim: int):
        """
        Args:
            in_channels (int): Number of input channels (C).
            kernel_size (int): Patch height/width used by unfold/fold.
            stride (int): Stride for patch extraction.
            padding (int): Replication padding applied to spatial dims before unfolding.
            hidden_dim (int): Dimension of the intermediate hidden projection for each patch.
        """
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.hidden_dim = hidden_dim

        # Padding layer
        self.pad = nn.ReplicationPad2d(padding)

        # Nonlinearities
        self.softplus = nn.Softplus()
        self.softsign = nn.Softsign()

        # Learnable linear transforms for patch -> hidden
        # D = in_channels * kernel_size * kernel_size
        D = in_channels * kernel_size * kernel_size
        self.D = D

        # Weight matrices: initialized with small random values
        self.weight1 = nn.Parameter(torch.randn(D, hidden_dim) * (1.0 / (D ** 0.5)))
        self.bias1 = nn.Parameter(torch.zeros(hidden_dim))

        self.weight2 = nn.Parameter(torch.randn(D, hidden_dim) * (1.0 / (D ** 0.5)))
        self.bias2 = nn.Parameter(torch.zeros(hidden_dim))

        # Projection from hidden back to patch-space (for folding)
        self.weight_out = nn.Parameter(torch.randn(hidden_dim, D) * (1.0 / (hidden_dim ** 0.5)))
        self.bias_out = nn.Parameter(torch.zeros(D))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor with shape (N, C, H, W)

        Returns:
            torch.Tensor: Output tensor with shape (N, C, H, W) (same channels/spatial as input)
                          after the complex patch-based processing.
        """
        N, C, H, W = x.shape
        assert C == self.in_channels, f"Expected in_channels={self.in_channels}, got {C}"

        # 1) Pad spatial dimensions
        x_pad = self.pad(x)  # shape: (N, C, H + 2*pad, W + 2*pad)
        H_pad = H + 2 * self.padding
        W_pad = W + 2 * self.padding

        # 2) Extract patches: (N, C * ks * ks, L) where L is number of spatial locations
        patches = F.unfold(x_pad, kernel_size=self.kernel_size, stride=self.stride)  # (N, D, L)
        # Transpose to (N, L, D) for per-patch matmuls
        patches_t = patches.permute(0, 2, 1)  # (N, L, D)
        # Convert to float if necessary (keep dtype)
        # patches_t = patches_t.contiguous()

        # 3) Two parallel linear projections + nonlinearities
        # branch1: Softplus(W1^T * patch + b1)
        # branch2: Softsign(W2^T * patch + b2)
        # We use einsum for clarity: 'nld,dh->nlh'
        hidden1 = torch.einsum('nld,dh->nlh', patches_t, self.weight1) + self.bias1  # (N, L, hidden)
        hidden1 = self.softplus(hidden1)

        hidden2 = torch.einsum('nld,dh->nlh', patches_t, self.weight2) + self.bias2
        hidden2 = self.softsign(hidden2)

        # 4) Interaction / fusion:
        # Create a gating signal from hidden2 and apply it to hidden1, also mix symmetric component.
        gate = torch.sigmoid(hidden2)  # gate in (0,1)
        fused = hidden1 * gate + hidden2 * (1.0 - gate)  # (N, L, hidden)

        # Additionally incorporate a residual-style multiplicative modulation
        modulation = torch.tanh(hidden1.mean(dim=-1, keepdim=True))  # (N, L, 1)
        fused = fused * (1.0 + modulation)  # (N, L, hidden) broadcasted modulation

        # 5) Project fused features back to patch space (D) and fold to spatial tensor
        recon_patches = torch.einsum('nlh,hd->nld', fused, self.weight_out) + self.bias_out  # (N, L, D)
        # Transpose back to (N, D, L)
        recon_patches = recon_patches.permute(0, 2, 1).contiguous()  # (N, D, L)

        # 6) Fold patches back into spatial layout of padded image
        x_recon_pad = F.fold(recon_patches, output_size=(H_pad, W_pad), kernel_size=self.kernel_size, stride=self.stride)
        # F.fold sums overlapping contributions. To compensate for overlaps, compute overlap-count using ones
        ones = torch.ones((N, C, H_pad, W_pad), dtype=x.dtype, device=x.device)
        ones_patches = F.unfold(ones, kernel_size=self.kernel_size, stride=self.stride)
        overlap = F.fold(ones_patches, output_size=(H_pad, W_pad), kernel_size=self.kernel_size, stride=self.stride)
        # Avoid division by zero
        overlap = torch.clamp(overlap, min=1.0)
        x_recon_pad = x_recon_pad / overlap

        # 7) Crop padding to restore original spatial size
        if self.padding > 0:
            x_recon = x_recon_pad[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            x_recon = x_recon_pad

        # Final shape: (N, C, H, W)
        return x_recon


# Configuration variables
batch_size = 8
in_channels = 16
height = 64
width = 64
kernel_size = 3
stride = 2
padding = 1
hidden_dim = 128

def get_inputs():
    # Random input tensor matching the configuration above
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    # Initialization parameters for Model constructor
    return [in_channels, kernel_size, stride, padding, hidden_dim]