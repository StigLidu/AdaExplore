import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class Model(nn.Module):
    """
    A compact vision module that demonstrates a multi-step processing pipeline:
      1. Nearest-neighbor upsampling
      2. 3x3 convolution to mix spatial information
      3. ReLU6 activation
      4. Channel-wise dropout (Dropout2d)
      5. Global spatial pooling and a final linear projection to class logits

    This pattern combines nn.UpsamplingNearest2d, nn.Conv2d, nn.ReLU6, nn.Dropout2d,
    and nn.AdaptiveAvgPool2d to form a small but non-trivial computation graph.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        scale_factor: int = 2,
        dropout_p: float = 0.2
    ):
        """
        Initializes layers used in the model.

        Args:
            in_channels (int): Number of input channels.
            hidden_channels (int): Number of intermediate convolutional channels.
            num_classes (int): Number of output classes for the final linear layer.
            scale_factor (int): Nearest-neighbor upsampling factor.
            dropout_p (float): Probability for nn.Dropout2d.
        """
        super(Model, self).__init__()
        # Upsample using nearest neighbor to increase spatial resolution
        self.up = nn.UpsamplingNearest2d(scale_factor=scale_factor)

        # 3x3 convolution preserves spatial dims with padding=1
        self.conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=True)

        # Non-linearity capped at 6 for numerical stability on mobile/quantized targets
        self.relu6 = nn.ReLU6(inplace=True)

        # Dropout2d randomly zeros whole channels; useful for regularizing conv feature maps
        self.dropout2d = nn.Dropout2d(p=dropout_p)

        # Global pooling to collapse spatial dims to 1x1, followed by a linear classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(hidden_channels, num_classes)

        # Initialize classifier bias to small values for stable initial outputs
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the pipeline.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, H, W)

        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes)
        """
        # 1) Upsample spatial dimensions by nearest neighbor interpolation
        x = self.up(x)

        # 2) Spatial mixing via a small convolution
        x = self.conv(x)

        # 3) Non-linearity
        x = self.relu6(x)

        # 4) Channel-wise dropout to regularize feature channels
        x = self.dropout2d(x)

        # 5) Global spatial aggregation and linear projection to logits
        x = self.global_pool(x)            # shape: (B, C, 1, 1)
        x = torch.flatten(x, 1)           # shape: (B, C)
        logits = self.classifier(x)       # shape: (B, num_classes)
        return logits

# Configuration / default inputs for testing the module
batch_size = 8
in_channels = 3
hidden_channels = 32
num_classes = 10
scale_factor = 2
dropout_p = 0.2
height = 32
width = 32

def get_inputs() -> List[torch.Tensor]:
    """
    Returns a list containing a single input tensor for the model:
    shape (batch_size, in_channels, height, width)
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs() -> List:
    """
    Returns initialization parameters in the order expected by Model.__init__:
    [in_channels, hidden_channels, num_classes, scale_factor, dropout_p]
    """
    return [in_channels, hidden_channels, num_classes, scale_factor, dropout_p]