import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    A small decoder-like block that demonstrates a mix of spatial padding,
    transposed convolutions (upsampling), gating, and a creative use of
    1D replication padding applied to a reshaped spatial tensor.

    Computation summary:
      1. ReplicationPad2d to extend spatial boundaries.
      2. Two ConvTranspose2d branches applied to the padded input:
         - up branch produces main upsampled features.
         - gate branch produces gating values that modulate the upsampled features.
      3. Element-wise gating: up_feat * sigmoid(gate_feat).
      4. Collapse the height dimension into channels to apply ReplicationPad1d
         across the width (treating width as a sequence dimension),
         then perform a reduction (max) across the padded sequence,
         expand back and restore spatial shape.
      5. A final ConvTranspose2d fuse layer to produce the desired output channels,
         with a residual-style addition of the upsampled features for richer gradients.
    """
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        output_padding: int = 0,
        pad2d: tuple = (1, 1, 1, 1),
        pad1d: int = 2
    ):
        """
        Initializes the module.

        Args:
            in_channels (int): Number of channels in the input.
            mid_channels (int): Number of internal/middle channels after upsampling.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size for ConvTranspose2d (default 4).
            stride (int): Stride for ConvTranspose2d (default 2) typically used for upsampling.
            padding (int): Padding for ConvTranspose2d.
            output_padding (int): Output padding for ConvTranspose2d.
            pad2d (tuple): ReplicationPad2d padding (left, right, top, bottom).
            pad1d (int): ReplicationPad1d padding applied to the sequence (left & right).
        """
        super(Model, self).__init__()

        # Spatial replication pad to reduce boundary artifacts before transposed conv
        self.pad2d = nn.ReplicationPad2d(pad2d)

        # Two transposed conv branches: main upsampling and gating
        self.up = nn.ConvTranspose2d(
            in_channels, mid_channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding, output_padding=output_padding, bias=True
        )
        self.gate = nn.ConvTranspose2d(
            in_channels, mid_channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding, output_padding=output_padding, bias=True
        )

        # A small fuse transposed conv to map mid_channels back to out_channels
        self.fuse = nn.ConvTranspose2d(
            mid_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=True
        )

        # 1D replication pad used after reshaping spatial dims into a sequence
        self.pad1d = nn.ReplicationPad1d(pad1d)

        # Activation
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor with shape (N, in_channels, H_in, W_in).

        Returns:
            torch.Tensor: Output tensor with shape (N, out_channels, H_out, W_out),
                          where H_out and W_out are determined by the ConvTranspose2d settings.
        """
        # 1) Replicate-pad spatial boundaries
        x_padded = self.pad2d(x)

        # 2) Two transpose-conv branches (upsample and gating)
        up_feat = self.up(x_padded)      # (N, mid_channels, H_up, W_up)
        gate_feat = self.gate(x_padded)  # (N, mid_channels, H_up, W_up)

        # 3) Gating mechanism
        gate = torch.sigmoid(gate_feat)
        gated = up_feat * gate
        gated = self.act(gated)

        # 4) Collapse height into channels to treat width as a sequence for 1D padding
        N, C, H, W = gated.shape
        # reshape to (N, C * H, W) so pad1d can be applied on width dimension
        seq = gated.contiguous().view(N, C * H, W)  # (N, C*H, W)
        seq_padded = self.pad1d(seq)                 # (N, C*H, W + 2*pad1d)

        # 5) Reduce across the padded width (sequence) using a max reduction to capture boundary-extended features
        # This produces a per-(channel*height) descriptor
        pooled, _ = torch.max(seq_padded, dim=2)     # (N, C*H)

        # 6) Expand the pooled descriptor back to the original width and reshape to spatial form
        expanded = pooled.unsqueeze(2).expand(-1, -1, W)   # (N, C*H, W)
        restored = expanded.contiguous().view(N, C, H, W) # (N, C, H, W)

        # 7) Fuse and add a residual-style connection
        fused = self.fuse(restored)
        # ensure fused and up_feat have broadcastable shapes for addition
        # If out_channels != mid_channels, addition is still okay via broadcasting only if shapes match channels.
        # We'll add a projected residual when channels differ by summing fused with a reduced up_feat projection.
        if fused.shape[1] == up_feat.shape[1]:
            out = fused + up_feat
        else:
            # simple channel projection of up_feat to match fused channels
            # use adaptive average pooling over channels -> make a small projection via mean and expand
            proj = up_feat.mean(dim=1, keepdim=True)          # (N,1,H,W)
            proj = proj.expand(-1, fused.shape[1], -1, -1)    # (N, out_channels, H, W)
            out = fused + proj

        # 8) Final activation
        out = torch.tanh(out)
        return out


# ----------------------
# Configuration variables
# ----------------------
batch_size = 4
in_channels = 32
mid_channels = 64
out_channels = 16

# Input spatial dimensions (note ConvTranspose2d will typically upsample these)
in_height = 16
in_width = 16

# ConvTranspose2d parameters (common pattern to double spatial dims)
kernel_size = 4
stride = 2
padding = 1
output_padding = 0

# Replication padding amounts
pad2d = (1, 1, 1, 1)  # left, right, top, bottom before transposed conv
pad1d = 2             # left and right padding for replication along width sequence

def get_inputs():
    """
    Returns a list with one input tensor matching the expected input shape.
    """
    x = torch.randn(batch_size, in_channels, in_height, in_width)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for constructing the Model.
    Order corresponds to Model.__init__ signature:
      in_channels, mid_channels, out_channels,
      kernel_size, stride, padding, output_padding, pad2d, pad1d
    """
    return [in_channels, mid_channels, out_channels,
            kernel_size, stride, padding, output_padding,
            pad2d, pad1d]