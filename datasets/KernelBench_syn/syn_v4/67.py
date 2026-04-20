import torch
import torch.nn as nn

# Configuration
batch_size = 8
in_channels = 3
height = 64
width = 64

conv_out_channels = 64    # output channels of first convolution
glu_hidden = 32           # hidden channels after GLU (actual conv2 will produce 2 * glu_hidden)
conv_kernel = 3
pool_kernel = 2
num_classes = 10

class Model(nn.Module):
    """
    Complex 2D vision-style module demonstrating the use of MaxPool2d with indices,
    MaxUnpool2d to partially invert the pooling, GLU for gated channel-wise nonlinearity,
    and LogSoftmax for probabilistic output.

    Computation graph (high-level):
      x -> Conv2d -> ReLU -> MaxPool2d(return_indices=True) -> MaxUnpool2d ->
           Conv2d -> ReLU -> GLU(dim=1) -> AdaptiveAvgPool2d(1x1) -> Linear -> LogSoftmax
    """
    def __init__(
        self,
        in_channels: int = in_channels,
        conv_channels: int = conv_out_channels,
        glu_hidden_dim: int = glu_hidden,
        conv_kernel_size: int = conv_kernel,
        pool_kernel_size: int = pool_kernel,
        num_classes: int = num_classes,
    ):
        super(Model, self).__init__()
        # First feature extractor
        self.conv1 = nn.Conv2d(in_channels, conv_channels, kernel_size=conv_kernel_size, padding=conv_kernel_size // 2)
        # MaxPool that returns indices needed for unpooling
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_kernel_size, return_indices=True)
        # Corresponding MaxUnpool2d to invert the pooling (partial inverse)
        self.unpool = nn.MaxUnpool2d(kernel_size=pool_kernel_size, stride=pool_kernel_size)
        # Second conv produces 2 * glu_hidden_dim channels so GLU can split into gated and candidate parts
        self.conv2 = nn.Conv2d(conv_channels, glu_hidden_dim * 2, kernel_size=3, padding=1)
        # GLU along the channel dimension
        self.glu = nn.GLU(dim=1)
        # Pool down to a single spatial location, then linear classifier
        self.adapt_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(glu_hidden_dim, num_classes)
        # LogSoftmax for output probabilities (log-space)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Log-probabilities of shape (B, num_classes).
        """
        # Initial convolution + nonlinearity
        feat = torch.relu(self.conv1(x))                # (B, conv_channels, H, W)

        # MaxPool returns both values and indices for unpool
        pooled, indices = self.pool(feat)               # (B, conv_channels, H/2, W/2), indices same shape as pooled

        # Unpool back to approximate original spatial layout using the indices
        # Provide output_size to ensure exact match with original feature map size
        unpooled = self.unpool(pooled, indices, output_size=feat.size())  # (B, conv_channels, H, W)

        # Second conv to prepare for GLU (produces 2*glu_hidden channels)
        gated_preact = torch.relu(self.conv2(unpooled)) # (B, 2*glu_hidden, H, W)

        # Apply GLU along channel dimension to get gated output (B, glu_hidden, H, W)
        gated = self.glu(gated_preact)

        # Global spatial aggregation
        pooled_spatial = self.adapt_pool(gated)         # (B, glu_hidden, 1, 1)
        flattened = pooled_spatial.view(pooled_spatial.size(0), -1)  # (B, glu_hidden)

        # Classification head
        logits = self.fc(flattened)                     # (B, num_classes)
        return self.log_softmax(logits)

def get_inputs():
    """
    Returns a list containing a single input tensor appropriate for the model.

    Shape: (batch_size, in_channels, height, width)
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters that match the Model signature:
    [in_channels, conv_channels, glu_hidden_dim, conv_kernel_size, pool_kernel_size, num_classes]
    """
    return [in_channels, conv_out_channels, glu_hidden, conv_kernel, pool_kernel, num_classes]