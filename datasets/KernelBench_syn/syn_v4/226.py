import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A moderately complex vision-inspired module that demonstrates:
    - Circular padding to emulate wrap-around spatial context
    - Stacked convolutions for feature extraction
    - Hard shrinkage non-linearity for sparse activation
    - Global pooling followed by a linear classifier
    - LogSoftmax for stable log-probabilities output

    This model is functionally distinct from simple normalization examples:
    it mixes padding, convolutional feature extraction, a non-standard
    elementwise shrinkage nonlinearity, and a probabilistic output layer.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        padding: int = 1,
        kernel_size: int = 3,
        hardshrink_lambda: float = 0.5
    ):
        """
        Initializes the model components.

        Args:
            in_channels (int): Number of input channels.
            hidden_channels (int): Number of channels in intermediate convs.
            num_classes (int): Number of output classes.
            padding (int): Circular padding size applied before convolutions.
            kernel_size (int): Kernel size for convolutions.
            hardshrink_lambda (float): Lambda parameter for nn.Hardshrink.
        """
        super(Model, self).__init__()
        # Circular padding to provide wrap-around spatial context
        self.pad = nn.CircularPad2d(padding)

        # First conv: use no internal padding because we pre-pad with CircularPad2d
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size, bias=True, padding=0)
        # Second conv: further mix features
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, bias=True, padding=0)

        # Hard shrinkage activation encourages sparsity in activations
        self.hardshrink = nn.Hardshrink(lambd=hardshrink_lambda)

        # Global feature aggregator
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier head
        self.linear = nn.Linear(hidden_channels, num_classes, bias=True)

        # Stable log-probabilities
        self.logsoftmax = nn.LogSoftmax(dim=1)

        # Small internal nonlinearity for gating
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, H, W).

        Returns:
            torch.Tensor: Log-probabilities tensor of shape (batch_size, num_classes).
        """
        # 1) Circularly pad to preserve wrap-around edges
        x = self.pad(x)                                  # -> (B, C, H + 2*pad, W + 2*pad)

        # 2) First convolution + ReLU
        x = self.conv1(x)                                # -> (B, hidden, H, W)
        x = self.relu(x)

        # 3) Second circular padding + convolution (preserve spatial dims)
        x = self.pad(x)
        x = self.conv2(x)                                # -> (B, hidden, H, W)

        # 4) Hard shrink to introduce sparsity in activations
        x = self.hardshrink(x)                           # -> (B, hidden, H, W)

        # 5) Residual-style short skip: add a smoothed version of early features
        # Use global average of the first conv activations as a cheap context and broadcast
        context = torch.mean(torch.abs(x), dim=(2, 3), keepdim=True)  # -> (B, hidden, 1, 1)
        x = x + 0.1 * context                            # broadcast-add: subtle contextual bias

        # 6) Global pooling and classification
        x = self.pool(x)                                 # -> (B, hidden, 1, 1)
        x = torch.flatten(x, 1)                          # -> (B, hidden)
        logits = self.linear(x)                          # -> (B, num_classes)

        # 7) Return log-softmaxed probabilities
        return self.logsoftmax(logits)

# Configuration / default inputs
batch_size = 8
in_channels = 3
hidden_channels = 48
num_classes = 10
height = 64
width = 64
kernel_size = 3
padding = 1
hardshrink_lambda = 0.7

def get_inputs():
    """
    Returns example input tensors for the forward pass.
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization arguments for the Model constructor in order.
    (in_channels, hidden_channels, num_classes, padding, kernel_size, hardshrink_lambda)
    """
    return [in_channels, hidden_channels, num_classes, padding, kernel_size, hardshrink_lambda]