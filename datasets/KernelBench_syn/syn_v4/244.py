import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex 2D feature extractor that combines a lazily-initialized convolution,
    ReLU6 activations, spatial pooling, Alpha Dropout and a final linear projection.

    The model is designed to accept an input tensor of shape (batch_size, in_channels, H, W).
    LazyConv2d allows the module to infer in_channels from the first input at runtime.
    """
    def __init__(self,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 dropout_p: float = 0.1,
                 bias: bool = True):
        """
        Initializes the model components.

        Args:
            out_channels (int): Number of output channels for the convolution.
            kernel_size (int): Size of the convolving kernel.
            stride (int): Stride of the convolution.
            padding (int): Zero-padding added to both sides of the input.
            dilation (int): Spacing between kernel elements.
            dropout_p (float): Probability for AlphaDropout.
            bias (bool): Whether to include a bias term in the convolution.
        """
        super(Model, self).__init__()
        # Lazily-initialized 2D convolution (in_channels inferred on first forward)
        self.conv = nn.LazyConv2d(out_channels=out_channels,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=padding,
                                  dilation=dilation,
                                  bias=bias)
        # Non-linear activation
        self.relu6 = nn.ReLU6()
        # Spatial pooling to reduce spatial dims to 1x1 regardless of input H/W
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Regularizing dropout for SELU-style activations (works well with self-normalizing nets)
        self.alpha_drop = nn.AlphaDropout(p=dropout_p)
        # Final projection in feature space
        self.fc = nn.Linear(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining the operations:
            conv -> relu6 -> adaptive avg pool -> flatten -> alpha dropout -> linear -> relu6

        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, out_channels).
        """
        # 1) Convolution (in_channels inferred on first call)
        x = self.conv(x)
        # 2) Non-linearity
        x = self.relu6(x)
        # 3) Spatial aggregation to (1,1)
        x = self.avgpool(x)
        # 4) Collapse spatial dims -> (B, C)
        x = torch.flatten(x, start_dim=1)
        # 5) Regularizing dropout
        x = self.alpha_drop(x)
        # 6) Final linear projection
        x = self.fc(x)
        # 7) Final activation to keep outputs bounded
        x = self.relu6(x)
        return x

# Configuration variables
batch_size = 8
in_channels = 3      # will be inferred by LazyConv2d from the input tensor
height = 64
width = 64

out_channels = 32
kernel_size = 3
stride = 2
padding = 1
dilation = 1
dropout_p = 0.1
bias = True

def get_inputs():
    """
    Returns a list containing a single input tensor for the model:
        x: Tensor of shape (batch_size, in_channels, height, width)
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters for the Model constructor in the same order:
        out_channels, kernel_size, stride, padding, dilation, dropout_p, bias
    """
    return [out_channels, kernel_size, stride, padding, dilation, dropout_p, bias]