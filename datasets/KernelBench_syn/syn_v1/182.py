import torch
import torch.nn as nn
from typing import List

class Model(nn.Module):
    """
    Complex model that combines Dropout2d -> AdaptiveMaxPool1d -> Linear -> LogSoftmax.
    The model accepts a 4D input (batch, channels, height, width), applies channel-wise dropout,
    collapses spatial dims to a 1D sequence per channel, performs adaptive max pooling on that
    1D sequence, then classifies using a fully-connected layer followed by LogSoftmax.
    """
    def __init__(self, in_channels: int, pool_output_size: int, num_classes: int, dropout_prob: float = 0.2):
        """
        Initializes the module components.

        Args:
            in_channels (int): Number of input channels.
            pool_output_size (int): Output size for AdaptiveMaxPool1d (per-channel sequence length after pooling).
            num_classes (int): Number of output classes (final linear output dimension).
            dropout_prob (float): Probability for Dropout2d.
        """
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.pool_output_size = pool_output_size
        self.num_classes = num_classes

        # Randomly zero out entire channels during training.
        self.dropout2d = nn.Dropout2d(p=dropout_prob)

        # Reduces a variable-length 1D sequence (flattened spatial dims) to a fixed length per channel.
        self.adaptive_mp1d = nn.AdaptiveMaxPool1d(output_size=pool_output_size)

        # Final classification layer: each sample maps flattened (channels * pool_output_size) -> num_classes
        self.fc = nn.Linear(in_channels * pool_output_size, num_classes)

        # Log probabilities across classes
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation.

        Steps:
            1. Apply Dropout2d to zero out entire channels randomly.
            2. Reshape spatial dims (H, W) into a single sequence length L = H * W per channel.
            3. Apply AdaptiveMaxPool1d to reduce L -> pool_output_size.
            4. Flatten per-sample and apply linear classification.
            5. Return log-probabilities with LogSoftmax.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_channels, H, W)

        Returns:
            torch.Tensor: Log-probabilities with shape (batch_size, num_classes)
        """
        # 1) Dropout over channels
        x = self.dropout2d(x)

        # 2) Collapse spatial dims H, W into a single length L = H * W for each channel.
        batch_size, c, h, w = x.shape
        # Safety check: ensure channels match expected
        if c != self.in_channels:
            raise ValueError(f"Expected input with {self.in_channels} channels, got {c}")

        x = x.view(batch_size, c, h * w)  # shape: (batch, channels, L)

        # 3) Adaptive max pooling along the length dimension to produce fixed-size sequences per channel.
        x = self.adaptive_mp1d(x)  # shape: (batch, channels, pool_output_size)

        # 4) Flatten channel and pooled-length dimensions for the linear layer.
        x = x.view(batch_size, -1)  # shape: (batch, channels * pool_output_size)

        # 5) Linear projection to class logits, then LogSoftmax for log-probabilities.
        logits = self.fc(x)  # shape: (batch, num_classes)
        log_probs = self.logsoftmax(logits)  # shape: (batch, num_classes)

        return log_probs

# Configuration variables
batch_size = 8
in_channels = 32
height = 64
width = 64
pool_output_size = 16
num_classes = 100
dropout_prob = 0.25

def get_inputs() -> List[torch.Tensor]:
    """
    Returns sample input tensors for the model.

    Output:
        [x] where x is a torch.Tensor of shape (batch_size, in_channels, height, width)
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs() -> List:
    """
    Returns the initialization parameters for the Model constructor in the same order.

    Output:
        [in_channels, pool_output_size, num_classes, dropout_prob]
    """
    return [in_channels, pool_output_size, num_classes, dropout_prob]