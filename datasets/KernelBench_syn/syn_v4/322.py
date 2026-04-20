import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex model that combines BatchNorm2d, AvgPool2d and AdaptiveAvgPool1d
    to produce a compact per-channel representation with a small learned gating mechanism.

    Computation steps (high-level):
    1. BatchNorm2d over input channels
    2. ReLU activation
    3. 2x2 AvgPool2d to reduce spatial resolution
    4. Collapse spatial dimensions and apply AdaptiveAvgPool1d to produce a fixed-length
       per-channel feature vector
    5. Flatten per-channel features and pass through two linear layers with GELU nonlinearity
       to produce a per-channel gating vector
    6. Apply sigmoid gating to the projected features and combine with a global per-channel
       average to produce final per-channel outputs of shape (batch_size, channels)
    """
    def __init__(self, in_channels: int, pooled_seq_len: int = 16, reduction: int = 4):
        """
        Args:
            in_channels (int): Number of input channels expected.
            pooled_seq_len (int): The output length from AdaptiveAvgPool1d for each channel.
            reduction (int): Factor to reduce channel*pooled_seq_len in the intermediate projection.
        """
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.pooled_seq_len = pooled_seq_len

        # Normalize across channels for 4D input
        self.bn = nn.BatchNorm2d(in_channels)

        # Spatial downsampling
        self.avgpool2d = nn.AvgPool2d(kernel_size=2, stride=2)

        # Adaptive pooling to a fixed-length 1D sequence per channel
        self.adaptive_pool1d = nn.AdaptiveAvgPool1d(output_size=pooled_seq_len)

        # Two-layer projection that computes a compact gating vector per channel
        flattened_dim = in_channels * pooled_seq_len
        hidden_dim = max(in_channels * reduction, 8)  # ensure reasonable hidden size

        # Project flattened per-sample features down and back to channels
        self.fc1 = nn.Linear(flattened_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, in_channels, bias=True)

        # Initialize fc layers with a controlled scheme to stabilize training / testing
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, channels) representing
                          a gated per-channel summary.
        """
        # Input normalization and non-linearity
        x_norm = self.bn(x)
        x_act = F.relu(x_norm)

        # Capture a global per-channel average (before aggressive spatial pooling)
        # shape: (B, C)
        global_chan_avg = x_act.mean(dim=(2, 3))

        # Spatial downsampling
        x_down = self.avgpool2d(x_act)  # (B, C, H', W')

        # Collapse spatial dims to a sequence per channel
        B, C, Hs, Ws = x_down.shape
        seq = x_down.view(B, C, Hs * Ws)  # (B, C, L)

        # Adaptive 1D pooling to fixed length per channel
        seq_pooled = self.adaptive_pool1d(seq)  # (B, C, pooled_seq_len)

        # Flatten per-sample, per-channel features into a single vector per sample
        flat = seq_pooled.view(B, -1)  # (B, C * pooled_seq_len)

        # Two-layer projection with GELU activation to compute gating logits
        h = self.fc1(flat)  # (B, hidden_dim)
        h = F.gelu(h)
        gate_logits = self.fc2(h)  # (B, C)

        # Convert logits to [0,1] gating values
        gate = torch.sigmoid(gate_logits)  # (B, C)

        # Combine gating values with global per-channel averages
        out = global_chan_avg * gate  # (B, C)

        return out

# Configuration variables
batch_size = 8
channels = 64
height = 32
width = 32
pooled_seq_len = 16
reduction = 4

def get_inputs():
    """
    Produces a single 4D input tensor suitable for the Model above.

    Returns:
        list: [x] where x has shape (batch_size, channels, height, width)
    """
    x = torch.randn(batch_size, channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for Model constructor.

    Returns:
        list: [in_channels, pooled_seq_len, reduction]
    """
    return [channels, pooled_seq_len, reduction]