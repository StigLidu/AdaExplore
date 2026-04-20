import torch
import torch.nn as nn
from typing import List, Tuple

class Model(nn.Module):
    """
    Complex model that combines a 3D transposed convolution, a projection to a sequence,
    iterative processing with an LSTMCell, and AlphaDropout before a final linear projection.

    Computation steps:
    1. Upsample/spatial-transform input with ConvTranspose3d.
    2. Apply ReLU activation and global average pooling over spatial dims to get per-channel descriptors.
    3. Project descriptors into a sequence (seq_len timesteps) of feature vectors.
    4. Process the sequence iteratively using an LSTMCell.
    5. Apply AlphaDropout to the final hidden state and project to output features.
    """
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        seq_len: int,
        lstm_input_size: int,
        hidden_size: int,
        out_features: int,
        kernel_size: Tuple[int, int, int] = (3, 3, 3),
        stride: Tuple[int, int, int] = (2, 2, 2),
        padding: Tuple[int, int, int] = (1, 1, 1),
        dropout_prob: float = 0.1
    ):
        """
        Initializes the composite module.

        Args:
            in_channels (int): Number of channels in the 3D input.
            mid_channels (int): Number of channels produced by the ConvTranspose3d.
            seq_len (int): Length of the sequence to generate for LSTM processing.
            lstm_input_size (int): Feature size for each LSTM timestep input.
            hidden_size (int): Hidden size of the LSTMCell.
            out_features (int): Dimensionality of the final output vector.
            kernel_size, stride, padding: ConvTranspose3d geometric params.
            dropout_prob (float): AlphaDropout probability.
        """
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.seq_len = seq_len
        self.lstm_input_size = lstm_input_size
        self.hidden_size = hidden_size
        self.out_features = out_features

        # Transposed convolution to increase spatial resolution / transform features
        self.convT = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )

        # Project pooled channel descriptors to a flattened sequence (seq_len * lstm_input_size)
        self.project_to_sequence = nn.Linear(mid_channels, seq_len * lstm_input_size)

        # LSTM cell for iterative sequence processing
        self.lstm_cell = nn.LSTMCell(input_size=lstm_input_size, hidden_size=hidden_size)

        # AlphaDropout applied to the final hidden state before output projection
        self.alpha_dropout = nn.AlphaDropout(p=dropout_prob)

        # Final projection to desired output dimensionality
        self.output_proj = nn.Linear(hidden_size, out_features)

        # Small initialization tweaks for stability
        nn.init.kaiming_normal_(self.convT.weight, nonlinearity='relu')
        if self.convT.bias is not None:
            nn.init.zeros_(self.convT.bias)
        nn.init.xavier_uniform_(self.project_to_sequence.weight)
        nn.init.zeros_(self.project_to_sequence.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features)
        """
        # 1) ConvTranspose3d to transform and upsample spatially
        x_conv = self.convT(x)                      # (B, mid_channels, D2, H2, W2)
        x_act = torch.relu(x_conv)                  # Non-linearity

        # 2) Global average pooling across spatial dims -> (B, mid_channels)
        pooled = x_act.mean(dim=(2, 3, 4))          # (B, mid_channels)

        # 3) Project to a flattened sequence and reshape -> (B, seq_len, lstm_input_size)
        seq_flat = self.project_to_sequence(pooled) # (B, seq_len * lstm_input_size)
        seq = seq_flat.view(seq_flat.size(0), self.seq_len, self.lstm_input_size)

        # Prepare initial hidden and cell states (zeros) with correct device/dtype
        batch_size = x.size(0)
        device = x.device
        dtype = x.dtype
        h = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        c = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)

        # 4) Iterate through the sequence using LSTMCell
        # We'll also add a residual-like skip from the pooled descriptor into each timestep input
        # to encourage temporal conditioning on global information.
        # Expand pooled descriptor to shape (B, lstm_input_size) via a simple linear-like repeat/truncate.
        # To avoid extra parameters, use a simple linear mapping via slicing/repeating:
        pooled_expanded = pooled
        if pooled_expanded.size(1) >= self.lstm_input_size:
            pooled_expanded = pooled_expanded[:, :self.lstm_input_size]
        else:
            repeats = (self.lstm_input_size + pooled_expanded.size(1) - 1) // pooled_expanded.size(1)
            pooled_expanded = pooled_expanded.repeat(1, repeats)[:, :self.lstm_input_size]
        # Iterate timesteps
        for t in range(self.seq_len):
            timestep_in = seq[:, t, :] + pooled_expanded * 0.1  # small conditioning residual
            h, c = self.lstm_cell(timestep_in, (h, c))

        # 5) Apply AlphaDropout to final hidden state and project to output
        h_dropped = self.alpha_dropout(h)
        out = self.output_proj(h_dropped)             # (B, out_features)
        return out

# Module-level configuration / default sizes
batch_size = 8
in_channels = 4
mid_channels = 16
depth = 8
height = 16
width = 16

seq_len = 10
lstm_input_size = 32
hidden_size = 64
out_features = 128

# ConvTranspose3d geometric params
kernel_size = (3, 3, 3)
stride = (2, 2, 2)
padding = (1, 1, 1)
dropout_prob = 0.1

def get_inputs() -> List[torch.Tensor]:
    """
    Returns example input tensors for the model.
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs() -> List:
    """
    Returns the initialization arguments for the Model constructor in order.
    """
    return [
        in_channels,
        mid_channels,
        seq_len,
        lstm_input_size,
        hidden_size,
        out_features,
        kernel_size,
        stride,
        padding,
        dropout_prob
    ]