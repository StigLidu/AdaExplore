import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex model that:
      - Applies a pointwise convolution to mix input channels.
      - Extracts overlapping patches using nn.Unfold.
      - Applies a learned linear transform to each patch (per-patch MLP).
      - Applies a Hardsigmoid nonlinearity to the transformed patches.
      - Reconstructs the feature map using nn.Fold.
      - Normalizes the reconstructed feature map with LazyInstanceNorm3d (requires 5D input).
      - Produces a single-channel spatial map by aggregating across channels.

    This combines convolution, unfold/fold patch processing, a learnable per-patch linear
    mapping, a lazy 3D instance normalization, and a Hardsigmoid activation.
    """
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        output_H: int,
        output_W: int
    ):
        """
        Initializes the model components.

        Args:
            in_channels (int): Number of input channels.
            mid_channels (int): Number of intermediate channels after pointwise conv.
            kernel_size (int): Patch kernel size for unfold/fold.
            stride (int): Stride used for unfold/fold.
            padding (int): Padding used for unfold/fold.
            output_H (int): Height of the output image (used by Fold).
            output_W (int): Width of the output image (used by Fold).
        """
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_H = output_H
        self.output_W = output_W

        # Pointwise conv to mix input channels into mid_channels
        self.conv1x1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=True)

        # Unfold/Fold for patch extraction and reconstruction
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        self.fold = nn.Fold(output_size=(output_H, output_W), kernel_size=kernel_size,
                            padding=padding, stride=stride)

        # Per-patch linear mapping: maps each flattened patch vector to another of same size
        flattened_patch_dim = mid_channels * (kernel_size * kernel_size)
        self.patch_mlp = nn.Linear(flattened_patch_dim, flattened_patch_dim, bias=True)

        # LazyInstanceNorm3d: will initialize num_features on first forward based on input channels
        self.inst_norm3d = nn.LazyInstanceNorm3d()  # num_features determined lazily

        # Activation
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_channels, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (batch, 1, H, W) - spatial map aggregated over channels.
        """
        # 1) Pointwise conv to change channel dimensionality
        y = self.conv1x1(x)  # (N, mid_channels, H, W)

        # 2) Extract overlapping patches
        patches = self.unfold(y)  # shape: (N, mid_channels * kH * kW, L)
        # Prepare for per-patch linear: (N, L, D)
        patches_t = patches.permute(0, 2, 1)

        # 3) Learned per-patch linear mapping (applied to each patch vector)
        patches_mapped = self.patch_mlp(patches_t)  # (N, L, D)

        # 4) Nonlinearity
        patches_mapped = self.hardsigmoid(patches_mapped)

        # 5) Restore shape to (N, D, L) for folding
        patches_mapped = patches_mapped.permute(0, 2, 1)

        # 6) Fold patches back to spatial feature map (sums overlaps)
        recon = self.fold(patches_mapped)  # (N, mid_channels, output_H, output_W)

        # 7) Prepare 5D tensor for LazyInstanceNorm3d: (N, C, D=1, H, W)
        recon_5d = recon.unsqueeze(2)  # (N, mid_channels, 1, H, W)

        # 8) Instance normalization (lazy initialization will set num_features = mid_channels)
        normed_5d = self.inst_norm3d(recon_5d)  # (N, mid_channels, 1, H, W)

        # 9) Collapse depth dimension and aggregate across channels to produce single-channel map
        normed = normed_5d.squeeze(2)  # (N, mid_channels, H, W)
        out = torch.mean(normed, dim=1, keepdim=True)  # (N, 1, H, W)

        return out

# Configuration variables
batch_size = 4
in_channels = 3
mid_channels = 16
H = 64
W = 64
kernel_size = 3
stride = 2
padding = 1

def get_inputs():
    """
    Returns example input tensors for the model.

    The model expects a single input:
      - x: Tensor of shape (batch_size, in_channels, H, W)
    """
    x = torch.randn(batch_size, in_channels, H, W)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters used to construct the Model instance.

    These correspond to the Model.__init__ signature:
      in_channels, mid_channels, kernel_size, stride, padding, output_H, output_W
    """
    return [in_channels, mid_channels, kernel_size, stride, padding, H, W]