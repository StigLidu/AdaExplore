import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex module that processes 3D volumetric input with a 3D convolution,
    merges the depth into the channel dimension, upsamples spatially with
    bilinear interpolation, and applies a per-spatial-location linear projection.

    Input shape: (N, C_in, D, H, W)
    Steps:
      1. 3D convolution -> (N, C_mid, D, H, W)
      2. ReLU activation
      3. Merge depth into channels -> (N, C_mid * D, H, W)
      4. 2D bilinear upsampling -> (N, C_mid * D, H_up, W_up)
      5. For each spatial location, apply a linear map from (C_mid * D) -> FC_out
         producing output shape (N, FC_out, H_up, W_up)
    """
    def __init__(self, in_channels: int, mid_channels: int, depth: int,
                 up_scale: int, fc_out: int, kernel_size: int = 3):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.depth = depth
        self.up_scale = up_scale
        self.fc_out = fc_out

        padding = kernel_size // 2
        # 3D convolution over (D, H, W)
        self.conv3d = nn.Conv3d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding)

        # 2D bilinear upsampling to operate on (H, W) after merging depth into channels
        self.upsample2d = nn.UpsamplingBilinear2d(scale_factor=up_scale)

        # Linear projection applied per spatial location after upsampling.
        # This is implemented as a single weight matrix applied to the channel vector at each (h, w).
        # Weight shape: (mid_channels * depth, fc_out)
        self.linear_weight = nn.Parameter(torch.randn(mid_channels * depth, fc_out))
        self.linear_bias = nn.Parameter(torch.randn(fc_out))

        # Small activation
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (N, C_in, D, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (N, FC_out, H*up_scale, W*up_scale)
        """
        # 1. 3D convolution
        conv_out = self.conv3d(x)  # (N, C_mid, D, H, W)

        # 2. Activation
        activated = self.act(conv_out)

        # 3. Merge depth into channel dimension
        N, C_mid, D, H, W = activated.shape
        merged = activated.view(N, C_mid * D, H, W)  # (N, C_mid * D, H, W)

        # 4. 2D bilinear upsampling (operates on H, W)
        up = self.upsample2d(merged)  # (N, C_mid * D, H_up, W_up)

        # 5. For each spatial location, apply linear projection:
        #    rearrange to (N, H_up, W_up, C_feat) and perform matrix multiplication with weight
        #    result -> (N, H_up, W_up, FC_out) -> permute to (N, FC_out, H_up, W_up)
        up_perm = up.permute(0, 2, 3, 1)  # (N, H_up, W_up, C_feat)
        # matmul broadcasts over N, H_up, W_up
        projected = torch.matmul(up_perm, self.linear_weight) + self.linear_bias  # (N, H_up, W_up, FC_out)
        out = projected.permute(0, 3, 1, 2).contiguous()  # (N, FC_out, H_up, W_up)
        return out


# Configuration variables
N = 2          # batch size
C_IN = 3       # input channels
D = 5          # depth (number of slices)
H = 32         # height
W = 32         # width
C_MID = 8      # mid channels after Conv3d
UP_SCALE = 2   # spatial upsampling factor
FC_OUT = 64    # output channels after per-location linear projection
KERNEL_SIZE = 3

def get_inputs():
    """
    Returns a list with the primary input tensor expected by Model.forward:
        - x: torch.Tensor of shape (N, C_IN, D, H, W)
    """
    x = torch.randn(N, C_IN, D, H, W)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters intended to be passed to Model.__init__.
    """
    return [C_IN, C_MID, D, UP_SCALE, FC_OUT, KERNEL_SIZE]