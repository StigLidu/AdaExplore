import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex model combining 3D reflection padding, 2D max pooling applied per-depth-slice,
    and a lazy 1D transposed convolution. The pipeline is:

      Input (B, C, D, H, W)
        -> ReflectionPad3d (pads D,H,W)
        -> reshape to (B*D, C, H, W) and apply MaxPool2d on (H,W) per depth slice
        -> reshape back to (B, C, D, H2, W2)
        -> flatten spatial dims to (B, C, L) where L = D * H2 * W2
        -> LazyConvTranspose1d (in_channels inferred on first forward) producing (B, out_channels, L_out)
        -> ReLU activation

    This creates a nontrivial data-flow that mixes 3D padding, 2D spatial pooling across slices,
    and 1D sequence transposed-convolution expansion.
    """
    def __init__(self, out_channels: int, conv_kernel_size: int = 3, conv_stride: int = 2, pad: int = 1):
        """
        Args:
            out_channels (int): Number of output channels for the ConvTranspose1d layer.
            conv_kernel_size (int): Kernel size for the ConvTranspose1d.
            conv_stride (int): Stride for the ConvTranspose1d.
            pad (int): ReflectionPad3d padding (applied equally on all three spatial dims).
        """
        super(Model, self).__init__()
        # Reflection padding in 3D to increase each spatial dimension by 2*pad
        self.reflection_pad = nn.ReflectionPad3d(pad)
        # Apply 2D max pooling over H and W for each depth slice
        self.pool2d = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # Lazy ConvTranspose1d: out_channels first, in_channels determined at first forward call.
        # Use symmetric padding for the deconvolutional kernel
        self.deconv1d = nn.LazyConvTranspose1d(out_channels, kernel_size=conv_kernel_size,
                                               stride=conv_stride, padding=conv_kernel_size // 2)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor after the deconvolution and activation, shape (batch_size, out_channels, L_out).
        """
        # x: (B, C, D, H, W)
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input (B,C,D,H,W), got shape {tuple(x.shape)}")

        # 1) Reflection padding in all three spatial dims -> (B, C, D+2p, H+2p, W+2p)
        x = self.reflection_pad(x)

        B, C, Dp, Hp, Wp = x.shape

        # 2) Rearrange to (B*D, C, H, W) to apply 2D pooling independently per depth slice
        x_slices = x.transpose(1, 2).contiguous().view(B * Dp, C, Hp, Wp)
        pooled = self.pool2d(x_slices)  # (B*D, C, H2, W2)

        # 3) Restore to (B, C, D, H2, W2)
        H2 = pooled.shape[2]
        W2 = pooled.shape[3]
        pooled = pooled.view(B, Dp, C, H2, W2).transpose(1, 2)  # (B, C, D, H2, W2)

        # 4) Flatten spatial dims (D * H2 * W2) into a 1D sequence length L for ConvTranspose1d
        #    Resulting shape: (B, C, L)
        L = Dp * H2 * W2
        seq = pooled.reshape(B, C, L)

        # 5) Apply Lazy ConvTranspose1d (in_channels inferred on first call), producing (B, out_channels, L_out)
        out = self.deconv1d(seq)

        # 6) Non-linearity
        out = self.activation(out)

        return out

# Module-level configuration
batch_size = 8
in_channels = 3
depth = 4
height = 32
width = 32

# Initialization parameters for the model
pad = 1
out_channels = 16
conv_kernel_size = 5
conv_stride = 2

def get_inputs():
    """
    Returns:
        list: Single-element list containing a randomly initialized 5D tensor
              matching (batch_size, in_channels, depth, height, width).
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for the Model constructor in order:
    [out_channels, conv_kernel_size, conv_stride, pad]
    """
    return [out_channels, conv_kernel_size, conv_stride, pad]