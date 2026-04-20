import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex 3D -> 2D reduction model that demonstrates:
    - Lazy 3D convolution (nn.LazyConv3d) to avoid specifying in_channels at construction time.
    - Elementwise Softshrink nonlinearity (nn.Softshrink) applied to convolution outputs.
    - Adaptive 2D average pooling (nn.AdaptiveAvgPool2d) applied per depth slice by reshaping.
    - Depth-wise aggregation and channel-wise modulation using global statistics.
    The forward pass returns a flattened feature map per batch element.
    """
    def __init__(
        self,
        out_channels: int,
        kernel_size: int = 3,
        conv_stride: int = 2,
        conv_padding: int = 1,
        conv_dilation: int = 1,
        adaptive_output_size=(4, 4),
        softshrink_lambda: float = 0.5,
    ):
        """
        Initializes the components for the model.

        Args:
            out_channels (int): Number of output channels for the 3D convolution.
            kernel_size (int): Kernel size for the 3D convolution.
            conv_stride (int): Stride for the 3D convolution.
            conv_padding (int): Padding for the 3D convolution.
            conv_dilation (int): Dilation for the 3D convolution.
            adaptive_output_size (tuple): Output (H, W) size for AdaptiveAvgPool2d.
            softshrink_lambda (float): Lambda parameter for Softshrink.
        """
        super(Model, self).__init__()
        # Lazy 3D conv: in_channels will be inferred on the first forward pass
        self.conv3d = nn.LazyConv3d(
            out_channels,
            kernel_size=kernel_size,
            stride=conv_stride,
            padding=conv_padding,
            dilation=conv_dilation,
            bias=True,
        )
        # Adaptive 2D average pooling to reduce spatial HxW to a fixed size per depth-slice
        self.adaptive_pool2d = nn.AdaptiveAvgPool2d(adaptive_output_size)
        # Softshrink nonlinearity applied element-wise
        self.softshrink = nn.Softshrink(lambd=softshrink_lambda)
        # small epsilon for numerical stability if needed
        self.eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation steps:
          1. 3D convolution: (N, C_in, D, H, W) -> (N, C_out, D', H', W')
          2. Softshrink activation applied element-wise.
          3. Reshape to treat each depth slice as a separate 2D sample: (N*D', C_out, H', W')
          4. AdaptiveAvgPool2d applied to each depth slice.
          5. Restore depth dimension and aggregate across depth with mean.
          6. Channel-wise modulation using a sigmoid of channel means.
          7. Flatten per-batch features to produce final output: (N, C_out * out_h * out_w)

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width)

        Returns:
            torch.Tensor: Flattened output tensor of shape (batch_size, C_out * out_h * out_w)
        """
        # Step 1: 3D convolution (lazy initialization will infer in_channels on first call)
        y = self.conv3d(x)  # shape: (N, C_out, D2, H2, W2)
        # Step 2: Non-linear shrinkage
        y = self.softshrink(y)  # element-wise

        N, C_out, D2, H2, W2 = y.shape

        # Step 3: Move depth into batch dimension so AdaptiveAvgPool2d can operate
        # Current: (N, C_out, D2, H2, W2) -> permute -> (N, D2, C_out, H2, W2) -> view -> (N*D2, C_out, H2, W2)
        y_slices = y.permute(0, 2, 1, 3, 4).contiguous().view(N * D2, C_out, H2, W2)

        # Step 4: Adaptive average pooling per depth-slice
        p = self.adaptive_pool2d(y_slices)  # shape: (N*D2, C_out, out_h, out_w)
        out_h, out_w = p.shape[-2], p.shape[-1]

        # Step 5: Restore depth and aggregate across depth dimension
        p = p.view(N, D2, C_out, out_h, out_w)            # (N, D2, C_out, out_h, out_w)
        p = p.permute(0, 2, 1, 3, 4).contiguous()         # (N, C_out, D2, out_h, out_w)
        # Aggregate across depth -> (N, C_out, out_h, out_w)
        z = p.mean(dim=2)  # mean over depth dimension

        # Step 6: Channel-wise modulation: compute channel means and pass through sigmoid
        # channel_stats: (N, C_out, 1, 1)
        channel_stats = z.mean(dim=[2, 3], keepdim=True)
        modulator = torch.sigmoid(channel_stats)  # in (0,1)
        z_modulated = z * (modulator + self.eps)  # broadcast and modulate

        # Step 7: Flatten per-batch
        out = z_modulated.view(N, -1)  # (N, C_out * out_h * out_w)
        return out

# Module-level configuration variables
batch_size = 8
in_channels = 3  # will be inferred by LazyConv3d; used only to build input tensor
depth = 16
height = 64
width = 64

# Convolution and pooling configuration
out_channels = 32
kernel_size = 3
conv_stride = 2
conv_padding = 1
conv_dilation = 1
adaptive_output_size = (4, 4)
softshrink_lambda = 0.75

def get_inputs():
    """
    Returns a list containing a single input tensor shaped for 3D convolution:
    (batch_size, in_channels, depth, height, width)
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters required to construct the Model instance.
    Order corresponds to the Model.__init__ signature.
    """
    return [
        out_channels,
        kernel_size,
        conv_stride,
        conv_padding,
        conv_dilation,
        adaptive_output_size,
        softshrink_lambda,
    ]