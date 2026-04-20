import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex 3D-to-2D processing module that:
    - Applies circular padding in 3D (depth, height, width)
    - Flattens the depth dimension into the batch to run a 2D convolution (LazyConv2d)
      which will infer in_channels on first forward
    - Applies a ReLU non-linearity followed by Softmax2d across channels at each spatial location
    - Restores the depth dimension and aggregates across depth with a mean
    - Normalizes the final 4D tensor with a Frobenius-like norm per sample

    This pattern demonstrates mixing a 3D padding primitive with 2D convolutional
    processing by reshaping the tensor layout, and uses both lazy initialization
    and spatial softmax in a nontrivial pipeline.
    """
    def __init__(
        self,
        out_channels: int,
        conv_kernel: int,
        conv_stride: int,
        conv_padding: int,
        pad3d: tuple
    ):
        """
        Args:
            out_channels (int): Number of output channels for the 2D convolution.
            conv_kernel (int): Kernel size for the 2D convolution.
            conv_stride (int): Stride for the 2D convolution.
            conv_padding (int): Padding for the 2D convolution.
            pad3d (tuple): 6-tuple padding for CircularPad3d (l, r, t, b, f, bk).
        """
        super(Model, self).__init__()
        # Circular padding in 3D (expects input shaped N, C, D, H, W)
        self.pad3d = nn.CircularPad3d(pad3d)
        # LazyConv2d will infer in_channels the first time forward is called
        self.conv2d = nn.LazyConv2d(out_channels=out_channels,
                                    kernel_size=conv_kernel,
                                    stride=conv_stride,
                                    padding=conv_padding)
        # Non-linearities and spatial softmax over channels at each spatial location
        self.relu = nn.ReLU()
        self.softmax2d = nn.Softmax2d()
        # Small epsilon to stabilize normalization
        self.eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Input:
            x: Tensor with shape (N, C_in, D, H, W)

        Returns:
            Tensor with shape (N, out_channels, H_out, W_out) after aggregating depth
            and applying normalization per-sample.
        """
        # 1) Circularly pad in depth, height, and width
        #    Result: (N, C_in, D_p, H_p, W_p)
        x_padded = self.pad3d(x)

        N, C_in, Dp, Hp, Wp = x_padded.shape

        # 2) Move depth next to batch dimension to apply 2D conv to each depth slice:
        #    Permute to (N, Dp, C_in, Hp, Wp) then reshape -> (N*Dp, C_in, Hp, Wp)
        x_slices = x_padded.permute(0, 2, 1, 3, 4).reshape(N * Dp, C_in, Hp, Wp)

        # 3) Apply 2D convolution (LazyConv2d will infer in_channels here), then ReLU
        conv_out = self.conv2d(x_slices)                # (N*Dp, out_ch, H_out, W_out)
        conv_out = self.relu(conv_out)

        # 4) Apply spatial softmax across channels at every spatial location
        conv_out = self.softmax2d(conv_out)             # (N*Dp, out_ch, H_out, W_out)

        # 5) Restore depth grouping: (N, Dp, out_ch, H_out, W_out)
        out_ch = conv_out.shape[1]
        H_out = conv_out.shape[2]
        W_out = conv_out.shape[3]
        conv_out = conv_out.view(N, Dp, out_ch, H_out, W_out)

        # 6) Aggregate across the depth dimension (simple mean across D)
        #    Result: (N, out_ch, H_out, W_out)
        depth_agg = conv_out.mean(dim=1)

        # 7) Per-sample Frobenius-like normalization across (C,H,W)
        #    Compute sqrt(sum(square)) across dims (1,2,3), keep dims for broadcasting
        norm = torch.sqrt(torch.sum(depth_agg * depth_agg, dim=(1, 2, 3), keepdim=True) + self.eps)
        normalized = depth_agg / norm

        return normalized

# Configuration variables
batch_size = 8
in_channels = 3
depth = 8
height = 64
width = 64

out_channels = 16
conv_kernel = 3
conv_stride = 2
conv_padding = 1
# CircularPad3d expects a 6-tuple: (left, right, top, bottom, front, back)
pad3d = (1, 1, 1, 1, 1, 1)

def get_inputs():
    """
    Returns:
        A list containing a single 5D input tensor shaped (batch_size, in_channels, depth, height, width).
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Returns:
        Initialization parameters for the Model constructor in the order expected:
        [out_channels, conv_kernel, conv_stride, conv_padding, pad3d]
    """
    return [out_channels, conv_kernel, conv_stride, conv_padding, pad3d]