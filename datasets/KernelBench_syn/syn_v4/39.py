import torch
import torch.nn as nn
import torch.nn.functional as F

# Configuration variables
BATCH_SIZE = 4
IN_CHANNELS = 3
DEPTH = 4
HEIGHT = 64
WIDTH = 64

# Desired output 3D layout
DEPTH_OUT = 2            # output depth slices
CHANNEL_OUT = 16         # channels per depth slice in output => total out channels = DEPTH_OUT * CHANNEL_OUT

# Upsampling parameters for the 2D transposed convolution
DECONV_KERNEL_SIZE = 4
DECONV_STRIDE = 2
DECONV_PADDING = 1

# Whether to attempt wrapping the model in DistributedDataParallel using torch.distributed
USE_DDP = False


class Model(nn.Module):
    """
    A moderately complex model that demonstrates:
      - 3D instance normalization on an input volume
      - collapsing the depth dimension into channels to operate with 2D LazyConvTranspose2d
      - a transposed convolution (lazy in_channels) to upsample spatial resolution
      - a residual skip connection derived from the input with a 1x1 Conv2d to match channels
      - reshaping back to a 5D tensor and applying a second InstanceNorm3d

    Input:
      x: (N, C_in, D_in, H_in, W_in)

    Output:
      y: (N, C_out, D_out, H_out, W_out) where:
        C_out = CHANNEL_OUT
        D_out = DEPTH_OUT
        H_out/W_out depend on the transposed conv upsampling (stride)
    """

    def __init__(
        self,
        in_channels: int,
        depth_in: int,
        channel_out: int,
        depth_out: int,
        deconv_kernel_size: int = DECONV_KERNEL_SIZE,
        deconv_stride: int = DECONV_STRIDE,
        deconv_padding: int = DECONV_PADDING,
    ):
        """
        Initialize the model.

        Args:
            in_channels: number of input channels (C_in)
            depth_in: number of input depth slices (D_in)
            channel_out: number of channels per depth slice in the output (C_out)
            depth_out: number of depth slices in the output (D_out)
            deconv_kernel_size, deconv_stride, deconv_padding: parameters for the 2D transposed conv
        """
        super(Model, self).__init__()

        self.in_channels = in_channels
        self.depth_in = depth_in
        self.channel_out = channel_out
        self.depth_out = depth_out
        self.out_channels_total = channel_out * depth_out

        # First InstanceNorm3d normalizes over (C_in) channels for 5D input
        self.instnorm1 = nn.InstanceNorm3d(num_features=in_channels, affine=False, track_running_stats=False)

        # LazyConvTranspose2d: in_channels will be inferred on the first forward pass.
        # It transforms the 2D collapsed representation (channels = C_in * D_in) into out_channels_total
        self.deconv = nn.LazyConvTranspose2d(
            out_channels=self.out_channels_total,
            kernel_size=deconv_kernel_size,
            stride=deconv_stride,
            padding=deconv_padding,
            bias=True,
        )

        # A small 1x1 conv to refine the deconv output (keeps channel count equal to out_channels_total)
        self.conv1x1 = nn.Conv2d(self.out_channels_total, self.out_channels_total, kernel_size=1, bias=True)

        # Skip projection: map original per-slice channels to channel_out (1x1 Conv2d)
        self.skip_conv = nn.Conv2d(in_channels, channel_out, kernel_size=1, bias=True)

        # Second InstanceNorm3d that will operate on the final output channels (channel_out)
        self.instnorm2 = nn.InstanceNorm3d(num_features=channel_out, affine=False, track_running_stats=False)

        # Non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (N, C_in, D_in, H_in, W_in)

        Returns:
            y: (N, C_out, D_out, H_out, W_out)
        """
        # Validate shape
        if x.dim() != 5:
            raise ValueError("Expected 5D input tensor (N, C, D, H, W)")

        N, C_in, D_in, H_in, W_in = x.shape

        # 1) Normalize the 3D input per-instance
        x_norm = self.instnorm1(x)  # (N, C_in, D_in, H_in, W_in)

        # Preserve a copy of normalized input for skip connection before collapsing depth
        x_for_skip = x_norm

        # 2) Collapse depth into channel dimension to create a 2D feature map
        #    permute to (N, D_in, C_in, H_in, W_in) then combine D and C into channels
        x_collapsed = x_norm.permute(0, 2, 1, 3, 4).contiguous()  # (N, D_in, C_in, H_in, W_in)
        x_2d = x_collapsed.view(N, D_in * C_in, H_in, W_in)       # (N, C_in * D_in, H_in, W_in)

        # 3) Apply LazyConvTranspose2d to upsample spatially and map to the desired total output channels
        y = self.deconv(x_2d)           # (N, out_channels_total, H_out, W_out)
        y = self.act(y)
        y = self.conv1x1(y)             # (N, out_channels_total, H_out, W_out)

        H_out, W_out = y.shape[-2], y.shape[-1]

        # 4) Build skip connection:
        #    - average along depth to make a 2D map (N, C_in, H_in, W_in)
        #    - project channels to channel_out
        #    - resize spatially to (H_out, W_out) and tile across depth_out groups to match channels grouping
        skip_2d = x_for_skip.mean(dim=2)          # (N, C_in, H_in, W_in)
        skip_2d = self.skip_conv(skip_2d)         # (N, channel_out, H_in, W_in)
        skip_2d = F.interpolate(skip_2d, size=(H_out, W_out), mode='bilinear', align_corners=False)  # (N, channel_out, H_out, W_out)

        # Tile the skip across depth_out groups so channels become (depth_out * channel_out)
        skip_tiled = skip_2d.unsqueeze(1).repeat(1, self.depth_out, 1, 1, 1)  # (N, depth_out, channel_out, H_out, W_out)
        skip_tiled = skip_tiled.view(N, self.out_channels_total, H_out, W_out)  # (N, out_channels_total, H_out, W_out)

        # 5) Add skip (residual) to the deconv output
        y = y + skip_tiled
        y = self.act(y)

        # 6) Reshape channels into (depth_out, channel_out) and permute to standard 5D (N, C_out, D_out, H_out, W_out)
        y = y.view(N, self.depth_out, self.channel_out, H_out, W_out)  # (N, depth_out, channel_out, H_out, W_out)
        y = y.permute(0, 2, 1, 3, 4).contiguous()                      # (N, channel_out, depth_out, H_out, W_out)

        # 7) Final normalization across the output channels (InstanceNorm3d expects num_features=channel_out)
        y = self.instnorm2(y)

        return y


def wrap_model_for_ddp(model: nn.Module) -> nn.Module:
    """
    Helper that wraps the provided model into DistributedDataParallel if
    torch.distributed has been initialized. If not initialized, returns the original model.

    Note: This function is provided to demonstrate integration with nn.parallel.DistributedDataParallel.
    Proper initialization of torch.distributed (init_process_group, correct device placement, etc.)
    must be handled by the user/script that uses this module.
    """
    if USE_DDP and torch.distributed.is_initialized():
        # Wrap and return
        return nn.parallel.DistributedDataParallel(model)
    return model


# Static sizes for creating sample inputs
BATCH = BATCH_SIZE
IN_CH = IN_CHANNELS
D_IN = DEPTH
H = HEIGHT
W = WIDTH
D_OUT = DEPTH_OUT
C_OUT = CHANNEL_OUT


def get_inputs():
    """
    Returns a single input tensor of shape (BATCH, IN_CHANNELS, DEPTH, HEIGHT, WIDTH).
    """
    x = torch.randn(BATCH, IN_CH, D_IN, H, W)
    return [x]


def get_init_inputs():
    """
    Returns the initialization parameters needed to construct the Model:
      [in_channels, depth_in, channel_out, depth_out]

    If USE_DDP is True, the caller is expected to initialize torch.distributed before wrapping.
    """
    return [IN_CH, D_IN, C_OUT, D_OUT]