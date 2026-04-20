import torch
import torch.nn as nn
from typing import List

"""
Complex sequence processing module combining:
- nn.RNNCell to process inputs step-wise
- nn.RReLU as a randomized activation applied to RNN hidden states
- nn.LPPool1d to perform power-average pooling over the time dimension

Structure follows the provided examples:
- Model class inheriting from nn.Module
- get_inputs() that returns example input tensors
- get_init_inputs() that returns initialization parameters
"""

# Configuration variables
batch_size = 8
seq_len = 20
input_size = 16
hidden_size = 32
channels = 12            # number of channels after linear projection (for pooling)
lp_norm = 2.0            # p for LPPool1d (e.g., 2 for L2 pooling)
pool_kernel = 3
pool_stride = 2
rrelu_lower = 0.125
rrelu_upper = 0.333

class Model(nn.Module):
    """
    Sequence model that:
    - Processes the input sequence with an RNNCell across time steps.
    - Applies randomized leaky ReLU (RReLU) to each hidden state.
    - Projects hidden states into a channel dimension and applies LPPool1d along time.
    - Reduces the pooled result by taking the max over channels to produce the final output.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        channels: int,
        lp_norm: float,
        pool_kernel: int,
        pool_stride: int,
        rrelu_lower: float = 0.125,
        rrelu_upper: float = 0.333
    ):
        """
        Args:
            input_size (int): Dimensionality of input features per time step.
            hidden_size (int): Hidden size for the RNNCell.
            channels (int): Number of channels to project hidden states into before pooling.
            lp_norm (float): p value for LPPool1d (power-average pooling).
            pool_kernel (int): Kernel size for LPPool1d.
            pool_stride (int): Stride for LPPool1d.
            rrelu_lower (float): Lower bound for RReLU noise sampling.
            rrelu_upper (float): Upper bound for RReLU noise sampling.
        """
        super(Model, self).__init__()
        # Core recurrent cell
        self.rnn_cell = nn.RNNCell(input_size=input_size, hidden_size=hidden_size, nonlinearity='tanh')
        # Randomized leaky ReLU activation
        self.rrelu = nn.RReLU(lower=rrelu_lower, upper=rrelu_upper)
        # Project hidden state to channel dimension to prepare for 1D pooling over time
        self.project = nn.Linear(hidden_size, channels)
        # LPPool1d expects input shape (batch, channels, length), performs power-average pooling
        self.lppool = nn.LPPool1d(norm_type=lp_norm, kernel_size=pool_kernel, stride=pool_stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, input_size)

        Returns:
            torch.Tensor: Tensor of shape (batch, pooled_length) after channel-wise max reduction
                          following LPPool1d along the time dimension.
        """
        batch, seq_len_in, feat = x.shape
        assert feat == self.rnn_cell.input_size, f"Expected input feature size {self.rnn_cell.input_size}, got {feat}"

        # Initialize hidden state to zeros
        h = x.new_zeros(batch, self.rnn_cell.hidden_size)

        # Collect projected, activated hidden states for each time step
        projected_states: List[torch.Tensor] = []

        for t in range(seq_len_in):
            xt = x[:, t, :]  # (batch, input_size)
            # RNN cell update
            h = self.rnn_cell(xt, h)  # (batch, hidden_size)
            # Randomized leaky ReLU applied element-wise
            h_act = self.rrelu(h)  # (batch, hidden_size)
            # Linear projection to channel dimension
            p = self.project(h_act)  # (batch, channels)
            projected_states.append(p)

        # Stack across time -> (seq_len, batch, channels) -> permute to (batch, channels, seq_len)
        stacked = torch.stack(projected_states, dim=0).permute(1, 2, 0)

        # Apply LPPool1d along the time (length) dimension
        pooled = self.lppool(stacked)  # (batch, channels, pooled_length)

        # Reduce across channels by taking max (could serve as an adaptive channel selection)
        out = torch.max(pooled, dim=1)[0]  # (batch, pooled_length)

        return out

def get_inputs():
    """
    Returns a list containing a single input tensor shaped (batch_size, seq_len, input_size).
    """
    x = torch.randn(batch_size, seq_len, input_size)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for the Model constructor in the following order:
    [input_size, hidden_size, channels, lp_norm, pool_kernel, pool_stride, rrelu_lower, rrelu_upper]
    """
    return [input_size, hidden_size, channels, lp_norm, pool_kernel, pool_stride, rrelu_lower, rrelu_upper]