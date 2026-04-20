import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex sequence processing module that combines Instance Normalization,
    a projected feed-forward with RMSNorm normalization, HardTanh activation,
    and a gated residual connection followed by temporal pooling.

    Input shape: (batch_size, seq_len, feature_dim)
    Output shape: (batch_size, feature_dim)
    """
    def __init__(self, feature_dim: int, hidden_dim: int, seq_len: int, inst_affine: bool = True, eps: float = 1e-8):
        """
        Initializes the model components.

        Args:
            feature_dim (int): Dimensionality of input features.
            hidden_dim (int): Dimensionality of the intermediate hidden projection.
            seq_len (int): Length of the input sequence (kept for reference).
            inst_affine (bool): Whether InstanceNorm1d has learnable affine parameters.
            eps (float): Epsilon value for RMSNorm numerical stability.
        """
        super(Model, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len

        # Normalize across features for each instance and time step sequence-wise
        # InstanceNorm1d expects (N, C, L) so we will transpose before/after usage.
        self.inst_norm = nn.InstanceNorm1d(num_features=feature_dim, affine=inst_affine)

        # Project input features to a hidden dimension
        self.proj_in = nn.Linear(feature_dim, hidden_dim, bias=True)

        # RMS normalization over the hidden dimension
        self.rms = nn.RMSNorm(hidden_dim, eps=eps, elementwise_affine=True)

        # Non-linearity with clipped range to encourage bounded activations
        self.act = nn.Hardtanh(min_val=-0.5, max_val=0.5)

        # Project back to feature dimension
        self.proj_out = nn.Linear(hidden_dim, feature_dim, bias=True)

        # Gating mechanism: produce a scalar gate per time step
        self.gate_proj = nn.Linear(hidden_dim, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation.

        Steps:
        1. InstanceNorm1d over feature channels per instance/time: transpose to (N, C, L)
        2. Project to hidden_dim with linear layer.
        3. Apply RMSNorm over hidden dim.
        4. Clip activations with Hardtanh.
        5. Compute gate from hidden state and project hidden back to feature_dim.
        6. Combine projected output with original input via gated residual.
        7. Temporal mean pooling across seq_len to produce final representation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, feature_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, feature_dim)
        """
        # Save original for residual (batch, seq_len, feature_dim)
        orig = x

        # InstanceNorm1d expects (N, C, L) -> transpose to (batch, feature_dim, seq_len)
        x_t = x.transpose(1, 2)
        x_norm = self.inst_norm(x_t)
        # Back to (batch, seq_len, feature_dim)
        x_norm = x_norm.transpose(1, 2)

        # Project to hidden dimension
        h = self.proj_in(x_norm)  # (batch, seq_len, hidden_dim)

        # RMS normalization over last dim (hidden_dim)
        h_norm = self.rms(h)

        # Non-linearity (clipped)
        h_act = self.act(h_norm)

        # Compute gate scalar per time step: shape (batch, seq_len, 1)
        gate = torch.sigmoid(self.gate_proj(h_act))

        # Project back to feature dimension
        out_proj = self.proj_out(h_act)  # (batch, seq_len, feature_dim)

        # Gated residual combination
        combined = orig + gate * out_proj  # broadcasting gate over feature_dim

        # Temporal pooling: mean over seq_len
        pooled = combined.mean(dim=1)  # (batch, feature_dim)

        return pooled

# Configuration / hyperparameters
batch_size = 8
seq_len = 128
feature_dim = 256
hidden_dim = 512
inst_affine = True
eps = 1e-8

def get_inputs():
    """
    Returns a list containing one input tensor with shape (batch_size, seq_len, feature_dim).
    """
    x = torch.randn(batch_size, seq_len, feature_dim)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for Model.__init__ in the same order:
    feature_dim, hidden_dim, seq_len, inst_affine, eps
    """
    return [feature_dim, hidden_dim, seq_len, inst_affine, eps]