import torch
import torch.nn as nn

"""
Complex module combining ReflectionPad2d, Unfold/Fold patch processing, and Softmax2d attention.
Performs learned per-patch transformation and reconstructs the image with overlap-averaging,
then applies a channel-wise softmax attention map (Softmax2d) to modulate the reconstructed features.
"""

# Configuration / shape parameters
batch_size = 8
in_channels = 16
height = 64
width = 64

# Patch / folding parameters
kernel_size = 3
stride = 2
padding = 1
dilation = 1

# Internal model width
hidden_dim = 128

class Model(nn.Module):
    """
    Model that:
    - Pads input with ReflectionPad2d
    - Extracts sliding patches via nn.Unfold
    - Applies a learned MLP to each patch (two linear layers with ReLU)
    - Reconstructs the image from patches with nn.Fold
    - Normalizes overlapping contributions by the overlap-count map
    - Applies a 1x1 Conv followed by Softmax2d to produce per-channel attention per spatial location
    - Modulates the reconstructed image with that attention and returns the cropped original-sized output
    """
    def __init__(self,
                 in_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 dilation: int,
                 hidden_dim: int,
                 height: int,
                 width: int):
        super(Model, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.hidden_dim = hidden_dim

        # Compute padded spatial size for folding/unfolding
        self.height = height
        self.width = width
        self.height_padded = self.height + 2 * self.padding
        self.width_padded = self.width + 2 * self.padding

        # Reflection padding (we pad explicitly so Unfold/Fold use padding=0)
        self.refpad = nn.ReflectionPad2d(self.padding)

        # Unfold and Fold to extract and combine patches
        self.unfold = nn.Unfold(kernel_size=self.kernel_size,
                                dilation=self.dilation,
                                padding=0,   # already reflected
                                stride=self.stride)
        # Fold needs to know the output (padded) spatial size
        self.fold = nn.Fold(output_size=(self.height_padded, self.width_padded),
                            kernel_size=self.kernel_size,
                            dilation=self.dilation,
                            padding=0,
                            stride=self.stride)

        # Linear MLP applied per patch:
        # Input dim = in_channels * kernel_size * kernel_size
        self.patch_size = self.in_channels * (self.kernel_size ** 2)
        self.linear1 = nn.Linear(self.patch_size, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.patch_size)

        # 1x1 conv to produce attention logits, same number of channels to multiply with folded output
        self.conv1x1 = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1)

        # Softmax over channels per spatial location
        self.softmax2d = nn.Softmax2d()

        # Small epsilon to avoid division by zero when normalizing overlaps
        self.eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor with shape (B, C, H, W), expected H = self.height, W = self.width

        Returns:
            Tensor of shape (B, C, H, W) - reconstructed and attention-modulated output cropped to original size
        """
        # 1) Reflection pad
        x_padded = self.refpad(x)  # (B, C, H_pad, W_pad)

        # 2) Extract patches via Unfold
        patches = self.unfold(x_padded)  # (B, C*ks*ks, L)
        B, _, L = patches.shape

        # 3) Prepare for per-patch MLP: (B, L, patch_size)
        patches_t = patches.permute(0, 2, 1)  # (B, L, patch_size)

        # 4) Apply MLP per patch
        hidden = self.linear1(patches_t)      # (B, L, hidden_dim)
        hidden = torch.relu(hidden)
        out_patches = self.linear2(hidden)    # (B, L, patch_size)

        # 5) Put back in (B, patch_size, L) for Fold
        out_patches = out_patches.permute(0, 2, 1)  # (B, patch_size, L)

        # 6) Fold to reconstruct padded image (summing overlapping contributions)
        folded = self.fold(out_patches)  # (B, C, H_pad, W_pad)

        # 7) Compute overlap counts by folding ones to get per-pixel contribution counts
        ones = torch.ones_like(out_patches)  # (B, patch_size, L)
        counts = self.fold(ones)  # (B, C, H_pad, W_pad)

        # 8) Normalize by counts to average overlapping contributions
        normalized = folded / (counts + self.eps)

        # 9) Produce per-channel attention logits and convert to per-location softmax
        attn_logits = self.conv1x1(normalized)  # (B, C, H_pad, W_pad)
        attn = self.softmax2d(attn_logits)      # Softmax across channels at each (h,w)

        # 10) Modulate normalized reconstruction by attention (element-wise)
        modulated = normalized * attn  # (B, C, H_pad, W_pad)

        # 11) Crop to original spatial size (remove reflection padding)
        if self.padding > 0:
            h0 = self.padding
            h1 = h0 + self.height
            w0 = self.padding
            w1 = w0 + self.width
            out = modulated[:, :, h0:h1, w0:w1]
        else:
            out = modulated

        return out


def get_inputs():
    """
    Returns sample input tensor(s) for the model.
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters in the order expected by Model.__init__:
    (in_channels, kernel_size, stride, padding, dilation, hidden_dim, height, width)
    """
    return [in_channels, kernel_size, stride, padding, dilation, hidden_dim, height, width]