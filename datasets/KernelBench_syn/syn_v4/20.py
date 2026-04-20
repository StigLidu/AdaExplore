import torch
import torch.nn as nn

"""
Complex module that:
- Accepts a 5D input tensor (N, C_in, D, H, W)
- Applies 3D replication padding
- Flattens the depth dimension into channels to run a 2D transposed convolution (ConvTranspose2d)
- Reshapes the conv output back into a 5D tensor with a new channel/depth layout
- Applies 3D channel-wise dropout (Dropout3d)
- Performs a learned channel mixing via einsum with a provided projection matrix

This module demonstrates interplay between 3D padding/dropout and 2D transposed convolution
by temporarily folding/unfolding the depth dimension. All shapes are fixed by top-level
configuration so the ConvTranspose2d can be constructed deterministically.
"""

# Configuration (module-level)
BATCH = 8
IN_CHANNELS = 16
DEPTH = 4
IN_HEIGHT = 64
IN_WIDTH = 64

# ConvTranspose2d configuration - the number of output channels must be divisible by DEPTH
DECONV_OUT_CHANNELS = 32  # must be multiple of DEPTH (here 32 % 4 == 0)
DECONV_KERNEL_SIZE = 3
DECONV_STRIDE = 2
DECONV_PADDING = 1
DECONV_OUTPUT_PADDING = 1

DROPOUT_P = 0.2

class Model(nn.Module):
    """
    Model combining ReplicationPad3d -> reshape -> ConvTranspose2d -> reshape -> Dropout3d -> channel-mix.
    Forward signature:
        forward(x: Tensor, channel_proj: Tensor) -> Tensor
    Where:
        x: Tensor of shape (N, C_in, D, H, W)
        channel_proj: Tensor of shape (C_out_per_depth, C_mixed) used to mix channel dimension
                      (produces output channels = C_mixed)
    Returns:
        Tensor of shape (N, C_mixed, D, H_out, W_out)
    """
    def __init__(
        self,
        in_channels: int,
        depth: int,
        deconv_out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        output_padding: int = 0,
        dropout_p: float = 0.0
    ):
        super(Model, self).__init__()

        # Validate divisibility so we can reshape conv outputs back to depth-aware 5D tensor
        if deconv_out_channels % depth != 0:
            raise ValueError("deconv_out_channels must be divisible by depth for reshape back to 5D.")

        self.in_channels = in_channels
        self.depth = depth
        self.deconv_out_channels = deconv_out_channels
        self.out_channels_per_depth = deconv_out_channels // depth

        # ReplicationPad3d expects padding as (padW_left, padW_right, padH_left, padH_right, padD_left, padD_right)
        # We pad height and width slightly to create non-trivial spatial change before deconv
        self.pad3d = nn.ReplicationPad3d((2, 2, 1, 1, 0, 0))

        # ConvTranspose2d will receive in_channels folded as (in_channels * depth)
        self.deconv2d = nn.ConvTranspose2d(
            in_channels * depth,
            deconv_out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=True
        )

        self.dropout3d = nn.Dropout3d(p=dropout_p)

        # A small activation for non-linearity
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, channel_proj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (N, C_in, D, H, W).
            channel_proj: Projection matrix of shape (C_out_per_depth, C_mixed).
                          It mixes the per-depth channel dimension to produce final channels.

        Returns:
            Tensor of shape (N, C_mixed, D, H_out, W_out).
        """
        # Step 1: 3D replication padding
        # x -> (N, C_in, D, H_p, W_p)
        x = self.pad3d(x)

        # Step 2: fold depth into channels so we can apply a 2D deconvolution
        N, C_in, D, H_p, W_p = x.shape
        # Safety check
        if C_in != self.in_channels or D != self.depth:
            raise ValueError(f"Expected input with channels={self.in_channels} and depth={self.depth}, "
                             f"but got channels={C_in}, depth={D}")
        x = x.reshape(N, C_in * D, H_p, W_p)  # -> (N, C_in * D, H_p, W_p)

        # Step 3: ConvTranspose2d to upsample spatially and alter channels
        x = self.deconv2d(x)  # -> (N, deconv_out_channels, H_out, W_out)
        x = self.relu(x)

        # Step 4: reshape back to 5D by splitting channels into (out_channels_per_depth, depth)
        N, Cout, H_out, W_out = x.shape
        if Cout != self.deconv_out_channels:
            raise RuntimeError("Unexpected conv output channels.")
        x = x.reshape(N, self.out_channels_per_depth, self.depth, H_out, W_out)  # -> (N, outCpd, D, H_out, W_out)

        # Step 5: Dropout3d across channels (keeps dims as 5D)
        x = self.dropout3d(x)

        # Step 6: channel mixing using provided projection matrix
        # Expect channel_proj shape (outCpd, C_mixed)
        if channel_proj.dim() != 2 or channel_proj.shape[0] != self.out_channels_per_depth:
            raise ValueError(f"channel_proj must be shape ({self.out_channels_per_depth}, C_mixed)")

        # einsum mixes the channel dimension: 'b c d h w, c e -> b e d h w'
        x = torch.einsum('bcdhw,ce->bedhw', x, channel_proj)

        return x

# Provide init configuration for constructing the Model externally
def get_init_inputs():
    """
    Returns the arguments required to initialize the Model:
        [in_channels, depth, deconv_out_channels, kernel_size, stride, padding, output_padding, dropout_p]
    """
    return [
        IN_CHANNELS,
        DEPTH,
        DECONV_OUT_CHANNELS,
        DECONV_KERNEL_SIZE,
        DECONV_STRIDE,
        DECONV_PADDING,
        DECONV_OUTPUT_PADDING,
        DROPOUT_P
    ]

def get_inputs():
    """
    Produces example input tensors for the forward method:
        x: random tensor of shape (BATCH, IN_CHANNELS, DEPTH, IN_HEIGHT, IN_WIDTH)
        channel_proj: random projection matrix of shape (out_channels_per_depth, out_channels_per_depth)
                     (Here we keep final mixed channels same as out_channels_per_depth for simplicity.)
    """
    x = torch.randn(BATCH, IN_CHANNELS, DEPTH, IN_HEIGHT, IN_WIDTH)

    out_channels_per_depth = DECONV_OUT_CHANNELS // DEPTH
    # We'll mix to the same number of output channels per depth (C_mixed == out_channels_per_depth)
    channel_proj = torch.randn(out_channels_per_depth, out_channels_per_depth)

    return [x, channel_proj]