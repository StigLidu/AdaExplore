import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# Module-level configuration
batch_size = 8
seq_len = 20
input_size = 32
hidden_size = 48
pad_left = 2
pad_right = 3
pad_value = -0.5  # constant value used for padding

class Model(nn.Module):
    """
    Complex sequence processor that:
    - Pads the temporal dimension using ConstantPad1d with a constant value.
    - Runs a per-timestep LSTMCell over the padded sequence.
    - Applies a learnable projection to the original inputs and fuses it (residual-style)
      with a CELU-activated LSTM hidden state at each timestep.
    - Returns outputs trimmed to the original (unpadded) sequence length.

    Constructor arguments:
        input_size (int): number of features per timestep in the input.
        hidden_size (int): hidden size for the LSTMCell and output projections.
        pad_left (int): number of time steps to pad on the left.
        pad_right (int): number of time steps to pad on the right.
        pad_value (float): constant value to use for padding.
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 pad_left: int = 1,
                 pad_right: int = 1,
                 pad_value: float = 0.0):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.pad_left = pad_left
        self.pad_right = pad_right
        self.pad_value = float(pad_value)

        # Layers / parameters
        self.lstm_cell = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
        # CELU activation layer
        self.celu = nn.CELU(alpha=1.0)
        # ConstantPad1d will be applied to (N, C, L) where C == input_size and L == seq_len
        self.pad = nn.ConstantPad1d((self.pad_left, self.pad_right), self.pad_value)

        # Learnable projection used for a residual-style fusion:
        # project input features into hidden_size space
        self.proj_weight = nn.Parameter(torch.randn(hidden_size, input_size) * (1.0 / (input_size ** 0.5)))
        self.proj_bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, input_size)

        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, hidden_size),
                          corresponding to the model outputs trimmed to the original sequence length.
        """
        if x.dim() != 3:
            raise ValueError(f"Expected input of shape (batch, seq_len, input_size), got {x.shape}")

        batch, seq_len_in, feat = x.shape
        if feat != self.input_size:
            raise ValueError(f"Expected input_size={self.input_size} but got feature dim {feat}")

        # Permute to (batch, channels=input_size, length=seq_len) for ConstantPad1d
        x_perm = x.permute(0, 2, 1)  # (batch, input_size, seq_len)
        # Pad temporal dimension with a constant value
        x_padded = self.pad(x_perm)  # (batch, input_size, seq_len + pad_left + pad_right)
        seq_len_padded = x_padded.size(2)

        # Permute back to (seq_len_padded, batch, input_size) to iterate time steps
        x_time_major = x_padded.permute(2, 0, 1)  # (L, N, C)

        # Initialize LSTM states
        device = x.device
        dtype = x.dtype
        h = torch.zeros(batch, self.hidden_size, device=device, dtype=dtype)
        c = torch.zeros(batch, self.hidden_size, device=device, dtype=dtype)

        outputs: List[torch.Tensor] = []

        # Iterate over padded timesteps
        for t in range(seq_len_padded):
            xt = x_time_major[t]  # (batch, input_size)

            # LSTMCell update
            h, c = self.lstm_cell(xt, (h, c))  # both (batch, hidden_size)

            # Non-linear activation
            h_act = self.celu(h)  # (batch, hidden_size)

            # Project original input xt into hidden space
            proj = torch.matmul(xt, self.proj_weight.t()) + self.proj_bias  # (batch, hidden_size)

            # Fuse activated hidden state with projected input (residual-like)
            fused = h_act + proj  # (batch, hidden_size)

            outputs.append(fused)

        # Stack to (L_padded, batch, hidden) and permute to (batch, L_padded, hidden)
        out_padded = torch.stack(outputs, dim=0).permute(1, 0, 2)

        # Trim the padded timesteps to match the original input sequence length.
        # Because we padded on the left, the original sequence starts at index pad_left
        start = self.pad_left
        end = start + seq_len_in
        out_trimmed = out_padded[:, start:end, :]  # (batch, seq_len_in, hidden_size)

        return out_trimmed

def get_inputs():
    """
    Returns a list of input tensors for the forward pass.
    The model expects a tensor of shape (batch_size, seq_len, input_size).
    """
    x = torch.randn(batch_size, seq_len, input_size)
    return [x]

def get_init_inputs():
    """
    Returns initialization arguments for the Model constructor.
    Order: input_size, hidden_size, pad_left, pad_right, pad_value
    """
    return [input_size, hidden_size, pad_left, pad_right, pad_value]