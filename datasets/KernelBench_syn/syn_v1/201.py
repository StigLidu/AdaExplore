import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex 1D processing module that:
      - Uses nn.Unfold (applied by treating the 1D signal as a 2D tensor with height=1)
      - Normalizes patch-feature channels with nn.LazyInstanceNorm1d
      - Applies a non-linear gating and projection using nn.ConvTranspose1d
    The forward pipeline:
      x (N, C, L) ->
        unsqueeze -> (N, C, 1, L) ->
        Unfold -> (N, C * patch_size, L_out) ->
        LazyInstanceNorm1d -> ReLU ->
        channel-wise gate (tanh over temporal mean) * activated ->
        ConvTranspose1d projection -> (N, proj_channels, L_out)
    """
    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        unfold_stride: int,
        unfold_padding: int,
        unfold_dilation: int,
        proj_channels: int,
        deconv_kernel: int,
        deconv_stride: int,
        deconv_padding: int,
        deconv_output_padding: int = 0
    ):
        super(Model, self).__init__()
        # Save parameters
        self.in_channels = in_channels
        self.patch_size = patch_size

        # Unfold configured to slide along width (treat height as 1)
        # kernel_size and stride/padding/dilation are tuples (height, width); height=1
        self.unfold = nn.Unfold(
            kernel_size=(1, patch_size),
            dilation=(1, unfold_dilation),
            padding=(0, unfold_padding),
            stride=(1, unfold_stride)
        )

        # Lazy instance norm: num_features will be inferred at first forward pass
        # The unfolded feature dimension will be in_channels * patch_size
        self.norm = nn.LazyInstanceNorm1d()

        # ConvTranspose1d expects input shaped (N, C_in, L_in).
        c_in_deconv = in_channels * patch_size
        self.deconv = nn.ConvTranspose1d(
            in_channels=c_in_deconv,
            out_channels=proj_channels,
            kernel_size=deconv_kernel,
            stride=deconv_stride,
            padding=deconv_padding,
            output_padding=deconv_output_padding
        )

        # Small non-linearity
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (N, C, L)
        Returns:
            Tensor of shape (N, proj_channels, L_out) where L_out depends on unfold/conv settings.
        """
        # Treat the 1D signal as a (N, C, 1, L) image so Unfold can extract 1D patches
        x_4d = x.unsqueeze(2)  # (N, C, 1, L)

        # Extract sliding patches -> (N, C * patch_size, L_out)
        patches = self.unfold(x_4d)

        # Normalize across 'channels' dimension (LazyInstanceNorm1d initializes num_features here)
        patches_norm = self.norm(patches)

        # Non-linear activation
        activated = self.act(patches_norm)

        # Compute a simple channel-wise gate from temporal statistics:
        # mean over temporal axis -> (N, C*patch_size, 1), then tanh to get gating in (-1,1)
        gate = torch.tanh(activated.mean(dim=2, keepdim=True))

        # Apply gating (broadcast along temporal dimension)
        gated = activated * gate

        # Project / recompose with ConvTranspose1d
        out = self.deconv(gated)  # shape (N, proj_channels, L_out_deconv)

        return out


# Module-level configuration variables (example sizes)
batch_size = 8
in_channels = 12
sequence_length = 1024

# Unfold parameters (patch extraction)
patch_size = 16
unfold_stride = 8
unfold_padding = 4
unfold_dilation = 1

# ConvTranspose projection parameters
proj_channels = 40
deconv_kernel = 3
deconv_stride = 1
deconv_padding = 1
deconv_output_padding = 0

def get_inputs():
    # Random input signal: (batch, channels, length)
    x = torch.randn(batch_size, in_channels, sequence_length)
    return [x]

def get_init_inputs():
    # Return the initialization parameters in the same order as Model.__init__
    return [
        in_channels,
        patch_size,
        unfold_stride,
        unfold_padding,
        unfold_dilation,
        proj_channels,
        deconv_kernel,
        deconv_stride,
        deconv_padding,
        deconv_output_padding
    ]