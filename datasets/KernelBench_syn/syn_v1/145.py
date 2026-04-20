import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex module that:
    - Applies 3D zero-padding to a 5D input (N, C, D, H, W)
    - Flattens (C, D, H) into a single channel dimension to form a 1D sequence over W
    - Applies 1D max-pooling (with indices) followed by a Softsign activation
    - Performs MaxUnpool1d to reconstruct the pooled signal back to the original sequence length
    - Reshapes back to 5D, fuses with the padded input, and reduces over the depth axis

    This combines ZeroPad3d, MaxPool1d (+ indices), MaxUnpool1d, and Softsign in a non-trivial pattern.
    """
    def __init__(self, padding: tuple, pool_kernel: int, pool_stride: int = None):
        """
        Args:
            padding (tuple): 6-tuple for ZeroPad3d (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
            pool_kernel (int): Kernel size for MaxPool1d / MaxUnpool1d along the W dimension after flattening.
            pool_stride (int, optional): Stride for pooling. If None, stride == pool_kernel.
        """
        super(Model, self).__init__()
        if pool_stride is None:
            pool_stride = pool_kernel

        # Zero-pad 3D spatial dims
        self.pad3d = nn.ZeroPad3d(padding)

        # 1D pooling/unpooling along the flattened width dimension
        # Keep indices from MaxPool1d to use with MaxUnpool1d
        self.pool1d = nn.MaxPool1d(kernel_size=pool_kernel, stride=pool_stride, return_indices=True)
        self.unpool1d = nn.MaxUnpool1d(kernel_size=pool_kernel, stride=pool_stride)

        # Non-linearity
        self.act = nn.Softsign()

        # Store padding tuple for shape calculations in forward if needed
        self._padding = tuple(padding)
        self._pool_kernel = pool_kernel
        self._pool_stride = pool_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Steps:
        1. Pad input to (N, C, Dp, Hp, Wp)
        2. Flatten (C, Dp, Hp) -> channels' to form (N, C', Wp)
        3. MaxPool1d (return indices), then Softsign activation
        4. MaxUnpool1d using saved indices to reconstruct to (N, C', Wp)
        5. Reshape back to (N, C, Dp, Hp, Wp)
        6. Element-wise add with padded input and average over depth -> (N, C, Hp, Wp)
        """
        # x: (N, C, D, H, W)
        x_p = self.pad3d(x)  # (N, C, Dp, Hp, Wp)
        # Ensure contiguous memory before reshaping
        x_p = x_p.contiguous()
        N, C, Dp, Hp, Wp = x_p.shape

        # Flatten C, Dp, Hp into a single channel dimension to form a 1D sequence over Wp
        # New shape: (N, C * Dp * Hp, Wp)
        seq_channels = C * Dp * Hp
        x_seq = x_p.view(N, seq_channels, Wp)

        # Save original sequence size for unpooling
        orig_seq_size = x_seq.size()

        # MaxPool1d with indices
        pooled, indices = self.pool1d(x_seq)  # pooled: (N, C', L'), indices: same shape as pooled

        # Non-linearity
        activated = self.act(pooled)

        # Unpool back to original sequence length using indices and original size
        # This reconstructs to (N, C', Wp)
        unpooled = self.unpool1d(activated, indices, output_size=orig_seq_size)

        # Reshape back to 5D: (N, C, Dp, Hp, Wp)
        out_5d = unpooled.view(N, C, Dp, Hp, Wp)

        # Fuse with padded input (residual connection) and then reduce across depth
        fused = out_5d + x_p  # element-wise addition
        # Reduce over the depth dimension to produce a 4D output (N, C, Hp, Wp)
        result = fused.mean(dim=2)

        return result

# Configuration / default sizes
batch_size = 4
channels = 8
depth = 4
height = 16
width = 64

# ZeroPad3d expects (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
padding = (1, 2, 1, 1, 0, 1)  # will increase W by 3, H by 2, D by 1

pool_kernel = 3
pool_stride = 2

def get_inputs():
    """
    Returns a list containing one input tensor shaped (batch_size, channels, depth, height, width).
    """
    x = torch.randn(batch_size, channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for the Model: padding tuple, pool_kernel, pool_stride.
    """
    return [padding, pool_kernel, pool_stride]