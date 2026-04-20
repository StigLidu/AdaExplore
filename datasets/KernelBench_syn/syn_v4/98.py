import torch
import torch.nn as nn
import torch.nn.functional as F

# Configuration
BATCH_SIZE = 4
IN_CHANNELS = 12
DEPTH = 8
HEIGHT = 10
WIDTH = 10

# We will use a ConvTranspose3d with stride=2 so output spatial sizes will double.
UPSAMPLE_STRIDE = (2, 2, 2)
KERNEL_SIZE = (3, 3, 3)
PADDING = (1, 1, 1)
OUTPUT_PADDING = (1, 1, 1)

class Model(nn.Module):
    """
    Complex 3D upsampling module that:
    - Computes a channel-wise temporal summary of the input (averaging over H and W),
    - Pads that 1D temporal summary (ConstantPad1d) to match the upsampled depth,
    - Applies Hardshrink to sparsify the modulation signal,
    - Uses ConvTranspose3d to upsample the full 3D tensor,
    - Modulates the upsampled tensor with the padded/sparsified summary (broadcasted over H/W),
    - Applies a residual gating and final non-linearity.
    """
    def __init__(self,
                 in_channels: int = IN_CHANNELS,
                 kernel_size=KERNEL_SIZE,
                 stride=UPSAMPLE_STRIDE,
                 padding=PADDING,
                 output_padding=OUTPUT_PADDING):
        super(Model, self).__init__()

        # Transposed convolution to upsample spatial dims (D,H,W)
        self.deconv = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=in_channels,  # keep channel count
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=True
        )

        # ConstantPad1d will pad the temporal length (depth dimension after averaging H/W)
        # We pad (left, right) = (0, DEPTH) so that length goes from DEPTH -> 2*DEPTH
        # (this matches the ConvTranspose3d stride=2 behavior in depth dimension)
        pad_right = DEPTH  # expect upsample factor 2 -> target length 2*DEPTH
        self.pad1d = nn.ConstantPad1d((0, pad_right), 0.0)

        # Hardshrink to sparsify the modulation coefficients
        self.shrink = nn.Hardshrink(lambd=0.6)

        # Small learnable scalar to control residual mixing
        self.res_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, D, H, W)

        Returns:
            Tensor of shape (B, C, D*2, H*2, W*2)
        """
        # 1) Compute channel-wise temporal summary by averaging over H and W
        #    Result shape: (B, C, D)
        summary = x.mean(dim=[3, 4])  # average over height and width

        # 2) Pad the temporal summary to match the upsampled depth (2 * D)
        #    Input to ConstantPad1d must have shape (N, C, L)
        summary_padded = self.pad1d(summary)  # shape: (B, C, 2*D)

        # 3) Sparsify modulation via Hardshrink
        modulation = self.shrink(summary_padded)  # (B, C, 2*D)

        # 4) Upsample the full 3D tensor
        up = self.deconv(x)  # (B, C, 2*D, 2*H, 2*W)

        # 5) Broadcast modulation over H and W and apply gating
        #    modulation -> (B, C, 2*D, 1, 1) to broadcast over spatial dims
        modulation = modulation.unsqueeze(-1).unsqueeze(-1)

        gated = up * (1.0 + modulation)  # simple residual-like gating

        # 6) Residual blend with learnable scale and final non-linearity
        out = torch.tanh(gated + self.res_scale * up)

        return out

def get_inputs():
    """
    Returns:
        List containing a single 5D input tensor of shape (BATCH_SIZE, IN_CHANNELS, DEPTH, HEIGHT, WIDTH)
    """
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, DEPTH, HEIGHT, WIDTH)
    return [x]

def get_init_inputs():
    """
    No special initialization inputs required for this module.
    """
    return []