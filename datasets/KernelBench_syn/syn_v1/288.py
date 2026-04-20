import torch
import torch.nn as nn
from typing import List, Any

class Model(nn.Module):
    """
    Complex 1D-tensor processing module that combines LazyInstanceNorm1d,
    Mish activation, Dropout1d, and a channel-wise adaptive gating mechanism.

    Expected input shape: (batch_size, channels, seq_len)
    """
    def __init__(self, dropout_p: float = 0.2, affine: bool = True, eps: float = 1e-5):
        """
        Initializes the model.

        Args:
            dropout_p (float): Probability of an element to be zeroed in Dropout1d.
            affine (bool): If True, the InstanceNorm1d will have learnable affine parameters.
            eps (float): A value added to the denominator for numerical stability in InstanceNorm.
        """
        super(Model, self).__init__()
        # LazyInstanceNorm1d allows us to not specify num_features up-front;
        # it will be determined on the first forward pass based on the input.
        self.inorm = nn.LazyInstanceNorm1d(eps=eps, affine=affine)
        self.activation = nn.Mish()
        self.dropout = nn.Dropout1d(p=dropout_p)
        self._dropout_p = dropout_p
        self._affine = affine
        self._eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining normalization, non-linearity, channel dropout,
        channel-wise adaptive gating, and a small residual connection.

        Steps:
        1. Instance normalization per channel (lazy initialized).
        2. Mish activation.
        3. Dropout1d to zero out full channels across the sequence dimension.
        4. Compute channel-wise statistics (mean over sequence length) and build a
           per-channel gate via softmax across channels.
        5. Scale activations with (1 + gate) to preserve base magnitude while allowing
           emphasis on selected channels.
        6. Add a small residual from the original input to stabilize gradients.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, L).

        Returns:
            torch.Tensor: Output tensor of shape (B, C, L).
        """
        # Normalize (lazy init will set num_features on first call)
        normed = self.inorm(x)                      # (B, C, L)
        # Non-linearity
        activated = self.activation(normed)         # (B, C, L)
        # Dropout across channels (zeros entire channels randomly)
        dropped = self.dropout(activated)           # (B, C, L)
        # Channel-wise pooled statistic (mean across the sequence dimension)
        channel_mean = dropped.mean(dim=2)          # (B, C)
        # Softmax across channels to create a distribution per batch element
        channel_gate = torch.softmax(channel_mean, dim=1).unsqueeze(2)  # (B, C, 1)
        # Apply multiplicative gating, adding 1.0 to keep base signal and allow amplification
        gated = dropped * (1.0 + channel_gate)     # (B, C, L)
        # Small residual connection from the original input (stabilizes training)
        residual = 0.1 * x                          # (B, C, L)
        out = gated + residual                      # (B, C, L)
        return out

# Module-level configuration variables
batch_size = 8
channels = 32
seq_len = 128
dropout_p = 0.25
affine_flag = True
eps_value = 1e-5

def get_inputs() -> List[torch.Tensor]:
    """
    Returns a list containing a sample input tensor matching the expected shape.

    Shape: (batch_size, channels, seq_len)
    Values: Standard normal random tensor.
    """
    x = torch.randn(batch_size, channels, seq_len)
    return [x]

def get_init_inputs() -> List[Any]:
    """
    Returns the initialization parameters for the Model constructor in order.

    Returns:
        [dropout_p, affine_flag, eps_value]
    """
    return [dropout_p, affine_flag, eps_value]