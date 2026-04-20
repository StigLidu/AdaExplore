import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex example model combining convolution, MaxPool2d with indices,
    Softmax2d-based channel gating, Hardsigmoid non-linearity, and MaxUnpool2d.
    The model also includes a 1x1 projection for residual connection when needed.

    Forward computation:
        x -> conv1 -> relu -> maxpool(return_indices=True) -> channel softmax ->
        gated = pooled * softmaxed -> hardsigmoid(gated) -> unpool(using indices) ->
        conv2 -> (optional skip projection) -> out
    """
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        pool_kernel: int,
        pool_stride: int,
        pool_padding: int = 0,
    ):
        """
        Initializes layers and helpers.

        Args:
            in_channels (int): Number of input channels.
            mid_channels (int): Channels after the first convolution and before pooling.
            out_channels (int): Number of output channels produced by final conv.
            pool_kernel (int): Kernel size for MaxPool2d / MaxUnpool2d.
            pool_stride (int): Stride for MaxPool2d / MaxUnpool2d.
            pool_padding (int): Padding for MaxPool2d.
        """
        super(Model, self).__init__()
        # First feature extractor
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        # MaxPool2d with indices so we can invert with MaxUnpool2d later
        self.maxpool = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride,
                                    padding=pool_padding, return_indices=True)

        # Channel-wise softmax over spatial locations (Softmax2d)
        self.softmax2d = nn.Softmax2d()

        # Non-linear gating after channel-wise softmax scaling
        self.hardsigmoid = nn.Hardsigmoid()

        # Unpool to invert the maxpool operation
        self.unpool = nn.MaxUnpool2d(kernel_size=pool_kernel, stride=pool_stride, padding=pool_padding)

        # Final projection convolution
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=True)

        # Optional 1x1 skip projection to match channels for residual connection
        if in_channels != out_channels:
            self.skip_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        else:
            self.skip_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the composed operations.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C_in, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (N, C_out, H, W)
        """
        # Initial convolution + activation
        feat = self.conv1(x)          # (N, mid_channels, H, W)
        feat = self.relu(feat)

        # Max pooling with indices for exact unpooling later
        pooled, indices = self.maxpool(feat)  # pooled: (N, mid_channels, Hp, Wp)

        # Compute channel-wise softmax across channel dimension for each spatial location
        # softmaxed has same shape as pooled and acts as a set of attention weights per channel
        soft = self.softmax2d(pooled)  # (N, mid_channels, Hp, Wp)

        # Apply channel gating: scale pooled features by the softmaxed attention
        gated = pooled * soft  # elementwise scaling

        # Non-linear squashing to keep gate values in a compact range
        gated = self.hardsigmoid(gated)

        # Unpool back to original spatial dimensions using the stored indices and the size of feat
        unpooled = self.unpool(gated, indices, output_size=feat.size())  # (N, mid_channels, H, W)

        # Final projection
        out = self.conv2(unpooled)  # (N, out_channels, H, W)

        # Optional residual connection with 1x1 projection if channel dims differ
        if self.skip_proj is not None:
            out = out + self.skip_proj(x)
        else:
            out = out + x  # direct residual when channels match

        return out

# Configuration / default parameters for input generation and initialization
batch_size = 8
in_channels = 16
mid_channels = 32
out_channels = 16  # set equal to in_channels to allow direct residual by default
height = 64
width = 64
pool_kernel = 2
pool_stride = 2
pool_padding = 0

def get_inputs():
    """
    Returns a list containing the input tensor to the model.
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters for the Model constructor in order:
    [in_channels, mid_channels, out_channels, pool_kernel, pool_stride, pool_padding]
    """
    return [in_channels, mid_channels, out_channels, pool_kernel, pool_stride, pool_padding]