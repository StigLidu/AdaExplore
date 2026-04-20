import torch
import torch.nn as nn

# Configuration variables
batch_size = 4
in_channels = 8
depth = 8
height = 32
width = 32

pad3d = 1               # replication padding applied to all sides (depth, height, width)
depth_pool = 2          # number of consecutive depth slices to merge into channel dim
conv_out_channels = 16
conv_kernel_size = 3
conv_stride = 1
conv_padding = 1        # padding for Conv2d

class Model(nn.Module):
    """
    Model that demonstrates a hybrid 3D-to-2D processing pipeline:
    1. Replication padding in 3D to expand spatial dimensions.
    2. Chunking the depth dimension into non-overlapping blocks and folding
       each block into the channel dimension to create 2D feature maps.
    3. Applying a 2D convolution over the resulting feature maps.
    4. CELU activation.
    5. Reassembling outputs and reducing across depth-chunks.

    Input shape: (batch, in_channels, depth, height, width)
    Output shape: (batch, conv_out_channels, H_out, W_out)
    """
    def __init__(
        self,
        in_channels: int,
        conv_out_channels: int,
        conv_kernel_size: int,
        conv_stride: int = 1,
        conv_padding: int = 0,
        pad3d: int = 0,
        depth_pool: int = 1,
    ):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.depth_pool = depth_pool

        # Replication padding in 3D (applies to depth, height, width)
        self.pad3d = nn.ReplicationPad3d(pad3d)

        # After folding depth_pool depth slices into the channel dimension,
        # the Conv2d input channels become in_channels * depth_pool.
        conv_in_channels = in_channels * depth_pool
        self.conv2d = nn.Conv2d(
            in_channels=conv_in_channels,
            out_channels=conv_out_channels,
            kernel_size=conv_kernel_size,
            stride=conv_stride,
            padding=conv_padding,
        )

        # Non-linear activation
        self.activation = nn.CELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Steps:
        - x: (B, C, D, H, W)
        - pad with ReplicationPad3d -> (B, C, D_p, H_p, W_p)
        - unfold depth into non-overlapping blocks of size depth_pool:
          -> (B, C, num_chunks, depth_pool, H_p, W_p)
        - permute and collapse to form 2D feature maps:
          -> (B * num_chunks, C * depth_pool, H_p, W_p)
        - Conv2d + CELU -> (B * num_chunks, out_channels, H_out, W_out)
        - reshape back and average across chunks -> (B, out_channels, H_out, W_out)
        """
        # Validate input channel dimension
        if x.size(1) != self.in_channels:
            raise ValueError(f"Expected input with {self.in_channels} channels, got {x.size(1)}")

        # 1) ReplicationPad3d
        x = self.pad3d(x)  # (B, C, D_p, H_p, W_p)

        B, C, Dp, Hp, Wp = x.shape

        # Ensure Dp is divisible by depth_pool
        if Dp % self.depth_pool != 0:
            # As a fallback, trim extra slices from the end to make it divisible.
            # This keeps the computation simple and deterministic.
            trim = Dp % self.depth_pool
            x = x[:, :, : Dp - trim, :, :]
            Dp = x.size(2)

        num_chunks = Dp // self.depth_pool

        # 2) Unfold depth into blocks and fold depth_pool into channel dimension
        # x.unfold -> (B, C, num_chunks, depth_pool, H, W)
        x_unfold = x.unfold(dimension=2, size=self.depth_pool, step=self.depth_pool)
        # Move chunk dimension next to batch for easier conv processing:
        # (B, num_chunks, C, depth_pool, H, W)
        x_unfold = x_unfold.permute(0, 2, 1, 3, 4, 5).contiguous()
        # Collapse (C, depth_pool) -> (C * depth_pool) and (B, num_chunks) -> expanded batch
        B_nc = B * num_chunks
        x_2d = x_unfold.view(B_nc, C * self.depth_pool, Hp, Wp)  # (B * num_chunks, C * depth_pool, H, W)

        # 3) Conv2d
        out = self.conv2d(x_2d)  # (B * num_chunks, out_channels, H_out, W_out)

        # 4) Activation
        out = self.activation(out)

        # 5) Restore chunk dimension and reduce across chunks
        out_channels = out.size(1)
        H_out = out.size(2)
        W_out = out.size(3)
        out = out.view(B, num_chunks, out_channels, H_out, W_out)  # (B, num_chunks, out_channels, H_out, W_out)

        # Aggregate across depth-chunks (mean)
        out = out.mean(dim=1)  # (B, out_channels, H_out, W_out)

        return out


def get_inputs():
    """
    Returns a list containing a single 5D input tensor shaped according to the
    module-level configuration variables.
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization arguments for the Model constructor in the expected order.
    """
    return [in_channels, conv_out_channels, conv_kernel_size, conv_stride, conv_padding, pad3d, depth_pool]