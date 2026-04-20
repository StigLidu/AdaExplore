import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex 1D sequence model that demonstrates a small convolutional residual block using lazy initialization,
    parametric ReLU activations, and channel-wise dropout. The model accepts input tensors of shape (B, C_in, L)
    and produces class logits of shape (B, num_classes).

    Computation pattern:
      1. Lazy 1D convolution to project input channels -> hidden_channels (conv1)
      2. PReLU activation (per-channel)
      3. Dropout1d (channel-wise dropout)
      4. 1x1 convolution to mix channel information (conv_proj)
      5. PReLU activation
      6. Lazy 1x1 convolution on the residual path to match channels
      7. Residual addition
      8. Global average pooling over the sequence length
      9. Linear classifier to produce logits
    """
    def __init__(self, hidden_channels: int, kernel_size: int, dropout_p: float, num_classes: int):
        """
        Args:
            hidden_channels (int): Number of channels produced by the first lazy conv and used throughout the block.
            kernel_size (int): Kernel size for the first convolution (will use padding=kernel_size//2 to preserve length).
            dropout_p (float): Dropout probability for nn.Dropout1d.
            num_classes (int): Number of output classes for the final classifier.
        """
        super(Model, self).__init__()
        # Lazily infer input channels for the main convolution and for the skip/identity projection.
        # Using LazyConv1d allows the model to accept inputs with different input channel counts without explicit in_channels.
        self.conv1 = nn.LazyConv1d(out_channels=hidden_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.skip_conv = nn.LazyConv1d(out_channels=hidden_channels, kernel_size=1)

        # A small bottleneck/projection after the main conv to mix channels (non-lazy since input = hidden_channels).
        self.conv_proj = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1)

        # Parametric ReLU activations with one parameter per channel for flexible activations.
        self.prelu1 = nn.PReLU(num_parameters=hidden_channels)
        self.prelu2 = nn.PReLU(num_parameters=hidden_channels)

        # Channel-wise dropout
        self.dropout = nn.Dropout1d(p=dropout_p)

        # Final classifier after global pooling
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, L).

        Returns:
            torch.Tensor: Logits tensor of shape (B, num_classes).
        """
        # Preserve residual
        residual = x

        # Main path
        out = self.conv1(x)             # (B, hidden_channels, L)
        out = self.prelu1(out)         # (B, hidden_channels, L)
        out = self.dropout(out)        # (B, hidden_channels, L)
        out = self.conv_proj(out)      # (B, hidden_channels, L)
        out = self.prelu2(out)         # (B, hidden_channels, L)

        # Residual projection to match channels if needed (lazy conv will initialize on first call)
        residual = self.skip_conv(residual)  # (B, hidden_channels, L)

        # Residual addition
        out = out + residual

        # Global average pooling over the length dimension
        pooled = out.mean(dim=2)       # (B, hidden_channels)

        # Classification head
        logits = self.classifier(pooled)  # (B, num_classes)
        return logits

# Configuration variables
batch_size = 8
in_channels = 3
hidden_channels = 64
seq_len = 1024
kernel_size = 15
dropout_p = 0.2
num_classes = 10

def get_inputs():
    """
    Returns a list of input tensors for the model forward.
    Shape: (batch_size, in_channels, seq_len)
    """
    x = torch.randn(batch_size, in_channels, seq_len)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters for the Model:
    [hidden_channels, kernel_size, dropout_p, num_classes]
    """
    return [hidden_channels, kernel_size, dropout_p, num_classes]