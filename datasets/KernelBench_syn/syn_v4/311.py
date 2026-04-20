import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

"""
Complex PyTorch kernel module combining RNNCell, MaxUnpool3d, and AvgPool1d.

Computation pattern:
1. MaxPool3d with return_indices to obtain pooled features and indices.
2. Flatten pooled spatial dims and apply AvgPool1d to create a temporal sequence.
3. Process the sequence with an nn.RNNCell iteratively to produce a final hidden state.
4. Map the final hidden state to channel-wise modulation factors.
5. Modulate pooled features and reconstruct the original volume via MaxUnpool3d.

Module-level configuration variables are used so get_inputs() can create consistent tensors.
"""

# Configuration (tuneable)
BATCH_SIZE = 4
CHANNELS = 8
DEPTH = 16
HEIGHT = 16
WIDTH = 16

# Pooling configuration
POOL_KERNEL = (2, 2, 2)   # kernel_size for MaxPool3d / MaxUnpool3d
POOL_STRIDE = (2, 2, 2)   # stride for pooling

# Sequence / RNN configuration
SEQ_LEN = 16              # length of sequence produced from pooled spatial features
RNN_HIDDEN_SIZE = 32      # hidden size for the RNNCell

class Model(nn.Module):
    """
    Model that demonstrates a hybrid volumetric + sequential computation:
    - MaxPool3d -> AvgPool1d to produce a sequence -> RNNCell over time steps
    - Use final hidden state to modulate pooled features
    - Unpool back to original spatial resolution with MaxUnpool3d
    """
    def __init__(self, seq_len: int = SEQ_LEN, hidden_size: int = RNN_HIDDEN_SIZE,
                 pool_kernel: tuple = POOL_KERNEL, pool_stride: tuple = POOL_STRIDE,
                 channels: int = CHANNELS, depth: int = DEPTH, height: int = HEIGHT, width: int = WIDTH):
        """
        Initializes the model and its submodules.

        Args:
            seq_len: Number of time steps for the sequence derived from pooled spatial features.
            hidden_size: Hidden size for the RNNCell.
            pool_kernel: 3D pooling kernel size.
            pool_stride: 3D pooling stride.
            channels: Number of channels in the input volume.
            depth, height, width: Spatial dimensions of the input volume (used to compute pooling output size).
        """
        super(Model, self).__init__()

        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.pool_kernel = pool_kernel
        self.pool_stride = pool_stride
        self.channels = channels

        # MaxPool3d to produce indices for MaxUnpool3d
        self.pool3d = nn.MaxPool3d(kernel_size=self.pool_kernel, stride=self.pool_stride, return_indices=True)
        # MaxUnpool3d to reconstruct spatial resolution
        self.unpool3d = nn.MaxUnpool3d(kernel_size=self.pool_kernel, stride=self.pool_stride)

        # Compute pooled spatial dimensions to configure AvgPool1d kernel size
        d_out = (depth - pool_kernel[0]) // pool_stride[0] + 1
        h_out = (height - pool_kernel[1]) // pool_stride[1] + 1
        w_out = (width - pool_kernel[2]) // pool_stride[2] + 1
        self.pooled_spatial_len = d_out * h_out * w_out

        # Ensure we can split pooled spatial length into seq_len segments evenly
        assert self.pooled_spatial_len % self.seq_len == 0, (
            f"Pooled spatial length ({self.pooled_spatial_len}) must be divisible by seq_len ({self.seq_len})."
        )
        avgpool_kernel_size = self.pooled_spatial_len // self.seq_len

        # AvgPool1d expects input of shape (N, C, L). We'll pool L -> seq_len.
        self.avgpool1d = nn.AvgPool1d(kernel_size=avgpool_kernel_size, stride=avgpool_kernel_size)

        # RNNCell that processes one time-step at a time
        self.rnn_cell = nn.RNNCell(input_size=self.channels, hidden_size=self.hidden_size, nonlinearity='tanh')

        # Linear layers:
        # - to initialize hidden state from pooled features
        # - to map final hidden state to channel-wise modulation
        self.init_hidden_linear = nn.Linear(self.channels, self.hidden_size)
        self.modulation_linear = nn.Linear(self.hidden_size, self.channels)

        # A small channel-wise normalization to stabilize modulation scale
        self.channel_norm = nn.BatchNorm3d(self.channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input volumetric tensor of shape (B, C, D, H, W)

        Returns:
            Reconstructed volume after modulation and unpooling: tensor of shape (B, C, D, H, W)
        """
        # 1) MaxPool3d to get pooled features and indices for unpooling
        pooled, indices = self.pool3d(x)  # pooled: (B, C, D', H', W')

        # 2) Flatten spatial dims and create a 1D signal per (B, C)
        B, C, Dp, Hp, Wp = pooled.shape
        flattened = pooled.view(B, C, -1)  # (B, C, L)

        # 3) AvgPool1d over L to produce (B, C, seq_len)
        seq_feats = self.avgpool1d(flattened)  # (B, C, seq_len)
        # Reorder to (seq_len, B, C) for time-major iteration
        seq_feats = seq_feats.permute(2, 0, 1)  # (T, B, C)

        # 4) Initialize hidden state from global pooled statistics
        # Use mean across pooled spatial positions as a condensed descriptor (B, C)
        pooled_mean = flattened.mean(dim=2)  # (B, C)
        h = torch.tanh(self.init_hidden_linear(pooled_mean))  # (B, hidden_size)

        # 5) Iterate RNNCell over the sequence
        for t in range(self.seq_len):
            inp_t = seq_feats[t]  # (B, C)
            # RNNCell expects (B, input_size) -> input_size == C
            h = self.rnn_cell(inp_t, h)  # (B, hidden_size)

        # 6) Map final hidden state to channel-wise modulation and apply
        mod = torch.sigmoid(self.modulation_linear(h))  # (B, C) in (0,1)
        mod = mod.view(B, C, 1, 1, 1)  # broadcastable to pooled shape
        # Apply channel normalization then modulation (residual scaling)
        pooled_norm = self.channel_norm(pooled)
        pooled_mod = pooled_norm * (1.0 + (mod - 0.5))  # center modulation around 1.0

        # 7) Unpool back to original resolution
        # Provide output_size explicitly to ensure correct shape
        out = self.unpool3d(pooled_mod, indices, output_size=x.size())  # (B, C, D, H, W)

        return out

# Input generation sizes are derived from module-level configuration
B = BATCH_SIZE
C = CHANNELS
D = DEPTH
H = HEIGHT
W = WIDTH

def get_inputs() -> List[torch.Tensor]:
    """
    Create example input tensors consistent with the model configuration.

    Returns:
        A list containing a single 5D tensor: (BATCH_SIZE, CHANNELS, DEPTH, HEIGHT, WIDTH)
    """
    x = torch.randn(B, C, D, H, W)
    return [x]

def get_init_inputs() -> List:
    """
    Return initialization parameters for constructing the Model.

    Order matches Model.__init__ signature: seq_len, hidden_size, pool_kernel, pool_stride, channels, depth, height, width
    """
    return [SEQ_LEN, RNN_HIDDEN_SIZE, POOL_KERNEL, POOL_STRIDE, CHANNELS, DEPTH, HEIGHT, WIDTH]