import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex model that demonstrates a mixed 2D/3D processing pipeline:
    - Applies InstanceNorm2d to a 4D input (N, C, H, W)
    - Expands it to 5D by repeating along a depth axis
    - Applies ConstantPad3d to pad the 3D spatial volume
    - Performs MaxPool3d (with indices) and then MaxUnpool3d to reconstruct
    - Crops out the padded regions, collapses the depth via mean reduction
    - Applies a final 1x1 Conv2d for channel mixing

    This showcases interaction between InstanceNorm2d, ConstantPad3d, and MaxUnpool3d.
    """
    def __init__(
        self,
        channels: int,
        depth: int = 4,
        pool_kernel: tuple = (2, 2, 2),
        pad: tuple = (1, 1, 2, 2, 1, 1),
        conv_out_channels: int = None,
        pad_value: float = 0.0,
    ):
        """
        Args:
            channels (int): Number of channels in the input (C).
            depth (int): Depth to expand the 2D input into (D).
            pool_kernel (tuple): Kernel size (and stride) for MaxPool3d/MaxUnpool3d.
            pad (tuple): Padding for ConstantPad3d as (pad_left, pad_right,
                         pad_top, pad_bottom, pad_front, pad_back).
            conv_out_channels (int, optional): Output channels for final Conv2d.
                                               If None, uses same as input channels.
            pad_value (float): Constant value used in ConstantPad3d.
        """
        super(Model, self).__init__()
        self.channels = channels
        self.depth = depth
        self.pool_kernel = pool_kernel
        self.pad_vals = pad  # (left, right, top, bottom, front, back)
        self.pad_value = pad_value

        # Instance normalization across channels for 2D input
        self.inst_norm = nn.InstanceNorm2d(num_features=channels, affine=True)

        # MaxPool3d (returns indices) and corresponding MaxUnpool3d
        self.pool3d = nn.MaxPool3d(kernel_size=pool_kernel, stride=pool_kernel, return_indices=True)
        self.unpool3d = nn.MaxUnpool3d(kernel_size=pool_kernel, stride=pool_kernel)

        # ConstantPad3d pads (D, H, W): (left, right, top, bottom, front, back)
        self.pad3d = nn.ConstantPad3d(pad, value=pad_value)

        # Final 1x1 Conv2d to mix channels after collapsing depth dimension
        out_ch = conv_out_channels if conv_out_channels is not None else channels
        self.final_conv = nn.Conv2d(in_channels=channels, out_channels=out_ch, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the mixed pipeline.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (N, out_channels, H, W)
        """
        if x.dim() != 4:
            raise ValueError("Input must be a 4D tensor (N, C, H, W)")

        N, C, H, W = x.shape
        if C != self.channels:
            raise ValueError(f"Input channels ({C}) do not match model channels ({self.channels})")

        # 1) Instance Normalization (2D)
        x_norm = self.inst_norm(x)  # (N, C, H, W)

        # 2) Expand to 3D volume by creating a depth dimension and repeating
        #    Resulting shape: (N, C, D, H, W)
        x_3d = x_norm.unsqueeze(2).repeat(1, 1, self.depth, 1, 1)

        # 3) Pad the 3D volume with ConstantPad3d
        #    pad tuple ordering in nn.ConstantPad3d: (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
        x_padded = self.pad3d(x_3d)
        # Record padded dimensions for later slicing / unpool output_size
        padded_size = x_padded.size()

        # 4) MaxPool3d with indices, then MaxUnpool3d to reconstruct
        pooled, indices = self.pool3d(x_padded)  # pooled shape reduced by pool_kernel
        # Unpool back to padded shape. Pass output_size explicitly for stable reconstruction.
        unpooled = self.unpool3d(pooled, indices, output_size=padded_size)

        # 5) Crop out the padding to recover original depth/height/width extents
        pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back = self.pad_vals
        d_start = pad_front
        d_end = pad_front + self.depth
        h_start = pad_top
        h_end = pad_top + H
        w_start = pad_left
        w_end = pad_left + W
        cropped = unpooled[:, :, d_start:d_end, h_start:h_end, w_start:w_end]  # (N, C, D, H, W)

        # 6) Collapse the depth dimension by averaging to return to 2D spatial shape
        collapsed = cropped.mean(dim=2)  # (N, C, H, W)

        # 7) Final 1x1 convolution for channel mixing and an activation
        out = self.final_conv(collapsed)
        out = F.relu(out, inplace=False)

        return out

# Configuration variables
batch_size = 8
channels = 3
height = 32
width = 32
depth = 4
pool_kernel = (2, 2, 2)
pad_values = (1, 1, 2, 2, 1, 1)  # (left, right, top, bottom, front, back)
pad_constant = 0.1
final_out_channels = 6

def get_inputs():
    """
    Generates a sample 4D input tensor for the model:
    - Shape: (batch_size, channels, height, width)
    """
    x = torch.randn(batch_size, channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters to construct the Model:
    [channels, depth, pool_kernel, pad_values, final_out_channels, pad_constant]
    """
    return [channels, depth, pool_kernel, pad_values, final_out_channels, pad_constant]