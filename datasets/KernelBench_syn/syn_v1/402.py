import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex 3D-to-2D processing module that:
      1. Applies 3D replication padding to a 5D input (N, C, D, H, W).
      2. Treats each depth slice as an independent 4D image by reshaping to (N*D, C, H, W).
      3. Upsamples each slice using nearest-neighbor 2D upsampling.
      4. Applies Instance Normalization per-channel on the upsampled slices.
      5. Reassembles the depth dimension and reduces across depth with an average to produce a 4D output (N, C, H_out, W_out).

    This creates a pipeline that mixes 3D padding, slice-wise 2D operations, normalization and a depth-aggregation stage.
    """
    def __init__(self, in_channels: int, pad: tuple, scale_factor: int, affine: bool = True):
        """
        Args:
            in_channels: Number of channels in the input tensor (C).
            pad: 6-element tuple for nn.ReplicationPad3d (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back).
            scale_factor: Integer scale factor for 2D upsampling.
            affine: Whether InstanceNorm2d has affine parameters.
        """
        super(Model, self).__init__()
        # 3D replication padding: expects 6 ints
        self.pad3d = nn.ReplicationPad3d(pad)
        # 2D upsampling (nearest neighbor)
        self.upsample2d = nn.UpsamplingNearest2d(scale_factor=scale_factor)
        # InstanceNorm2d operates on (N, C, H, W) where C == in_channels
        self.inorm = nn.InstanceNorm2d(num_features=in_channels, affine=affine)
        # Keep config for potential use in forward/inspection
        self.in_channels = in_channels
        self.pad = pad
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (N, C, D, H, W).

        Returns:
            Tensor of shape (N, C, H_out, W_out) where H_out and W_out reflect padding and upsampling.
        """
        # x: (N, C, D, H, W)
        if x.dim() != 5:
            raise ValueError("Expected 5D input (N, C, D, H, W)")

        N, C, D, H, W = x.shape

        # 1) Replication pad in 3D -> (N, C, D_p, H_p, W_p)
        x_padded = self.pad3d(x)

        # 2) Reinterpret each depth slice as an independent image: (N, C, D_p, H_p, W_p) -> (N*D_p, C, H_p, W_p)
        # To do this safely, permute to (N, D_p, C, H_p, W_p) then reshape.
        Np, Cp, Dp, Hp, Wp = x_padded.shape  # Np == N, Cp == C
        x_slices = x_padded.permute(0, 2, 1, 3, 4).contiguous().view(Np * Dp, Cp, Hp, Wp)

        # 3) Upsample each slice with nearest neighbor interpolation (2D)
        x_up = self.upsample2d(x_slices)

        # 4) Instance normalization across each slice
        x_norm = self.inorm(x_up)

        # 5) Reassemble depth and reduce across depth with mean
        # x_norm shape: (N * D_p, C, H_up, W_up)
        H_up, W_up = x_norm.shape[-2], x_norm.shape[-1]
        x_reassembled = x_norm.view(Np, Dp, Cp, H_up, W_up).permute(0, 2, 1, 3, 4).contiguous()  # (N, C, D_p, H_up, W_up)

        # Aggregate across depth dimension (simple average) to produce final 4D output
        out = x_reassembled.mean(dim=2)  # (N, C, H_up, W_up)
        return out

# Configuration / shapes
BATCH = 4
CHANNELS = 32
DEPTH = 8
HEIGHT = 64
WIDTH = 48

# ReplicationPad3d expects 6 ints: (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
PAD = (1, 2, 2, 1, 0, 1)
SCALE_FACTOR = 2
AFFINE = True

def get_inputs():
    """
    Returns the input tensor(s) for the model's forward pass.
    Input shape: (BATCH, CHANNELS, DEPTH, HEIGHT, WIDTH)
    """
    x = torch.randn(BATCH, CHANNELS, DEPTH, HEIGHT, WIDTH)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for the Model constructor.
    Order: [in_channels, pad, scale_factor, affine]
    """
    return [CHANNELS, PAD, SCALE_FACTOR, AFFINE]