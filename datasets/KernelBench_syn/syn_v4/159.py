import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class Model(nn.Module):
    """
    A moderately complex convolutional module combining multiple Conv2d layers,
    a per-element LogSigmoid gating, a residual-style 1x1 projection, and a
    final LogSoftmax over class logits produced by a 1x1 convolution followed
    by spatial global average pooling.

    Computation graph:
      x -> conv1 -> LogSigmoid (gating) -> conv2  \
       \                                          + -> conv3 (1x1) -> global pool -> LogSoftmax
        -> skip_proj (1x1) ---------------------/
    """
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        num_classes: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1
    ):
        """
        Initializes convolutional layers and activation modules.

        Args:
            in_channels (int): Number of channels in the input tensor.
            mid_channels (int): Number of channels after the first convolution.
            out_channels (int): Number of channels after the second convolution.
            num_classes (int): Number of output classes for classification.
            kernel_size (int): Kernel size for conv1 and conv2. Defaults to 3.
            stride (int): Stride for conv1 and conv2. Defaults to 1.
            padding (int): Padding for conv1 and conv2. Defaults to 1.
        """
        super(Model, self).__init__()
        # First convolution followed by an elementwise LogSigmoid gating
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.logsigmoid = nn.LogSigmoid()  # element-wise gating nonlinearity

        # Second convolution that will be combined with a residual-style projection
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)

        # 1x1 projection to match input channels to out_channels for the skip connection
        self.skip_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        # Final 1x1 convolution to produce class logits per spatial location
        self.classifier_conv = nn.Conv2d(out_channels, num_classes, kernel_size=1, stride=1, padding=0)

        # LogSoftmax over the class dimension after spatial pooling
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing log-probabilities for each class.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, H, W)

        Returns:
            torch.Tensor: Log-probabilities of shape (batch_size, num_classes)
        """
        # conv1 -> elementwise LogSigmoid gating (note: LogSigmoid outputs negative values,
        # using exponent to get gating factor in (0,1) is possible, but here we multiply in log-space by converting)
        # For numerical stability and to keep the pattern simple, we'll exponentiate logsigmoid to obtain gating factors.
        h1 = self.conv1(x)                           # (B, mid, H, W)
        gate = torch.exp(self.logsigmoid(h1))       # gating factors in (0,1), same shape as h1
        gated = h1 * gate                           # gated activation (elementwise modulation)

        # conv2 processes the gated features
        h2 = self.conv2(gated)                      # (B, out, H, W)

        # skip projection from input and add (residual connection)
        skip = self.skip_proj(x)                    # (B, out, H, W)
        fused = h2 + skip                           # fused features

        # Final 1x1 conv to produce per-location class logits
        logits_map = self.classifier_conv(fused)    # (B, num_classes, H, W)

        # Global average pooling over spatial dims -> (B, num_classes)
        pooled = logits_map.mean(dim=(2, 3))        # spatial mean

        # LogSoftmax to obtain log-probabilities over classes
        log_probs = self.logsoftmax(pooled)         # (B, num_classes)

        return log_probs

# Configuration / default inputs
batch_size = 8
in_channels = 3
mid_channels = 16
out_channels = 32
num_classes = 10
height = 64
width = 64
kernel_size = 3
stride = 1
padding = 1

def get_inputs() -> List[torch.Tensor]:
    """
    Returns a list containing one input tensor suitable for the Model forward method.

    Shape: (batch_size, in_channels, height, width)
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs() -> List:
    """
    Returns the arguments to construct the Model:
      [in_channels, mid_channels, out_channels, num_classes, kernel_size, stride, padding]
    """
    return [in_channels, mid_channels, out_channels, num_classes, kernel_size, stride, padding]