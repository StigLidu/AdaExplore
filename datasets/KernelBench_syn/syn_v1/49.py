import torch
import torch.nn as nn

# Configuration
batch_size = 8
seq_len = 128
input_dim = 64
hidden_size = 128
num_layers = 2
lp_kernel = 3
lp_stride = 2
lp_p = 2  # Lp norm degree for LPPool1d

class Model(nn.Module):
    """
    Sequence model that:
    - Encodes an input sequence with a multi-layer GRU
    - Applies 1D Lp pooling across the temporal dimension of GRU outputs
    - Applies a Hardsigmoid non-linearity
    - Computes a data-dependent gate from the final GRU hidden state and uses it
      to modulate the pooled features, with a residual projection.
    Final output is a compact per-batch feature vector (batch, hidden_size).
    """
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        lp_kernel: int = 3,
        lp_stride: int = 2,
        lp_p: int = 2,
    ):
        super(Model, self).__init__()
        # Recurrent encoder
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,  # (seq_len, batch, features)
            bidirectional=False,
        )
        # 1D Lp pooling across the temporal dimension (treated as length)
        self.pool = nn.LPPool1d(norm_type=lp_p, kernel_size=lp_kernel, stride=lp_stride)
        # Elementwise non-linearity
        self.hardsigmoid = nn.Hardsigmoid()
        # Gating: produce a gate from the last GRU layer's hidden state
        self.gate = nn.Linear(hidden_size, hidden_size)
        # Residual projection to mix pooled features
        self.res_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch, hidden_size)
        """
        # Prepare for GRU: (seq_len, batch, input_dim)
        x_seq = x.permute(1, 0, 2)

        # GRU encoding
        # gru_out: (seq_len, batch, hidden_size)
        # h_n: (num_layers, batch, hidden_size)
        gru_out, h_n = self.gru(x_seq)

        # Move to (batch, channels, length) for LPPool1d: (batch, hidden_size, seq_len)
        features = gru_out.permute(1, 2, 0)

        # Lp pooling across the temporal dimension -> reduces length
        pooled = self.pool(features)  # (batch, hidden_size, L_pooled)

        # Non-linearity applied elementwise
        activated = self.hardsigmoid(pooled)  # (batch, hidden_size, L_pooled)

        # Aggregate temporally to obtain a vector representation
        pooled_vec = torch.mean(activated, dim=2)  # (batch, hidden_size)

        # Compute a gate from the last GRU layer hidden state
        last_hidden = h_n[-1]  # (batch, hidden_size)
        gate = self.hardsigmoid(self.gate(last_hidden))  # (batch, hidden_size)

        # Modulate pooled representation with gate and add a residual projection
        gated = pooled_vec * gate  # elementwise gating
        residual = self.res_proj(pooled_vec)
        out = gated + residual  # (batch, hidden_size)

        return out

def get_inputs():
    """
    Generates a random batch of sequences for testing.

    Returns:
        list: single element list containing the input tensor x of shape (batch, seq_len, input_dim)
    """
    x = torch.randn(batch_size, seq_len, input_dim)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters to construct the Model.

    Returns:
        list: [input_dim, hidden_size, num_layers, lp_kernel, lp_stride, lp_p]
    """
    return [input_dim, hidden_size, num_layers, lp_kernel, lp_stride, lp_p]