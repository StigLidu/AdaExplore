import torch
import torch.nn as nn
from typing import List, Any

"""
Complex 1D sequence processing module that combines circular padding,
convolutional feature extraction, SELU activation, Dropout1d channel dropout,
and a residual connection with global average pooling.

Structure:
- Model class inheriting from nn.Module
- __init__ accepts configuration for channels, kernel size, and dropout
- forward applies: CircularPad1d -> Conv1d -> SELU -> Dropout1d -> Conv1d -> Residual Add -> Global AvgPool
- get_inputs() returns a sample input tensor
- get_init_inputs() returns initialization parameters for Model
"""

# Module-level configuration (defaults)
batch_size = 8
in_channels = 16
hidden_channels = 32
out_channels = 16
seq_len = 1024
kernel_size = 7
dropout_prob = 0.2

class Model(nn.Module):
    """
    1D convolutional block with circular padding, SELU activation, Dropout1d,
    and a residual connection. Produces per-example feature vectors by
    global average pooling across the temporal (sequence) dimension.
    """
    def __init__(
        self,
        in_ch: int,
        hid_ch: int,
        out_ch: int,
        kernel_sz: int,
        dropout_p: float = 0.1,
    ):
        """
        Initializes the model components.

        Args:
            in_ch (int): Number of input channels.
            hid_ch (int): Number of hidden convolutional channels.
            out_ch (int): Number of output channels.
            kernel_sz (int): Kernel size for the main convolution.
            dropout_p (float): Dropout probability for Dropout1d.
        """
        super(Model, self).__init__()

        # Ensure kernel size is at least 1
        if kernel_sz < 1:
            raise ValueError("kernel_sz must be >= 1")

        pad_amount = kernel_sz // 2  # symmetric circular padding

        # Circular padding over the temporal (last) dimension
        self.pad = nn.CircularPad1d((pad_amount, pad_amount))

        # Convolutional feature extractor preserving temporal length
        self.conv1 = nn.Conv1d(in_ch, hid_ch, kernel_size=kernel_sz, bias=True)

        # SELU activation for self-normalizing activations
        self.selu = nn.SELU()

        # Dropout that zeros out entire channels (features) along temporal dimension
        self.dropout = nn.Dropout1d(p=dropout_p)

        # 1x1 convolution to project hidden features to desired output channels
        self.conv2 = nn.Conv1d(hid_ch, out_ch, kernel_size=1, bias=True)

        # Residual projection if input and output channels differ
        if in_ch != out_ch:
            self.proj = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=True)
        else:
            self.proj = nn.Identity()

        # Initialize weights with a stable scheme suited for SELU (LeCun normal)
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='linear')
        nn.init.zeros_(self.conv1.bias)
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='linear')
        nn.init.zeros_(self.conv2.bias)
        if isinstance(self.proj, nn.Conv1d):
            nn.init.kaiming_normal_(self.proj.weight, nonlinearity='linear')
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Steps:
            1. Circularly pad the input to allow wrap-around convolution.
            2. Apply a 1D convolution to extract local features.
            3. Apply SELU activation for non-linearity.
            4. Apply Dropout1d for channel-wise regularization.
            5. Project to output channels via a 1x1 convolution.
            6. Add a residual connection (projecting input if necessary).
            7. Global average pool over the temporal dimension to produce a per-sample vector.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_channels, seq_len)

        Returns:
            torch.Tensor: Output tensor of shape (batch, out_channels)
        """
        # 1) Circular padding
        x_padded = self.pad(x)

        # 2) Local convolution
        features = self.conv1(x_padded)

        # 3) SELU non-linearity
        activated = self.selu(features)

        # 4) Channel-wise dropout
        dropped = self.dropout(activated)

        # 5) Project to output channels (still preserves temporal length)
        projected = self.conv2(dropped)

        # 6) Residual connection (project input if channels differ)
        residual = self.proj(x)

        # Ensure residual and projected have the same temporal length
        # (they should, because conv1 with symmetric padding and conv2 with kernel=1 preserve length)
        out = projected + residual

        # 7) Global average pooling across the temporal dimension -> (batch, out_ch)
        out = out.mean(dim=2)

        return out

def get_inputs() -> List[torch.Tensor]:
    """
    Returns example input tensors for the model.

    The model expects an input of shape (batch_size, in_channels, seq_len).
    """
    x = torch.randn(batch_size, in_channels, seq_len)
    return [x]

def get_init_inputs() -> List[Any]:
    """
    Returns initialization parameters for the Model constructor.

    Order: [in_channels, hidden_channels, out_channels, kernel_size, dropout_prob]
    """
    return [in_channels, hidden_channels, out_channels, kernel_size, dropout_prob]