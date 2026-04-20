import torch
import torch.nn as nn

"""
Complex PyTorch kernel module demonstrating a gated residual transformer-style block
that combines LayerNorm, ReLU6, and Sigmoid in a multi-step computation.

Structure:
- Model: nn.Module with configurable dimensions.
- forward: Applies layer normalization, a feed-forward bottleneck, gating, and residual connection.
- get_inputs: Produces a test input tensor.
- get_init_inputs: Returns initialization parameters for constructing the Model.
"""

# Configuration variables
batch_size = 8
seq_len = 128
model_dim = 512
hidden_dim = 2048  # typical FFN expansion (4x)

class Model(nn.Module):
    """
    Gated Residual Block:
    - Applies LayerNorm to the input.
    - Projects to a larger hidden dimension, applies ReLU6 activation.
    - Projects back to model dimension.
    - Computes a separate gate vector (via LayerNorm + linear + Sigmoid).
    - Combines projected output with input via element-wise gated residual.
    - Applies final LayerNorm for stabilized outputs.
    """

    def __init__(self, model_dim: int, hidden_dim: int):
        """
        Initializes the gated residual block.

        Args:
            model_dim (int): Dimensionality of input features.
            hidden_dim (int): Hidden intermediate dimensionality (usually > model_dim).
        """
        super(Model, self).__init__()
        # Normalization layers
        self.ln_in = nn.LayerNorm(model_dim)
        self.ln_gate = nn.LayerNorm(model_dim)
        self.ln_out = nn.LayerNorm(model_dim)

        # Feed-forward projection layers
        self.fc_expand = nn.Linear(model_dim, hidden_dim, bias=True)
        self.fc_reduce = nn.Linear(hidden_dim, model_dim, bias=True)

        # Gate projection: produces values in [0,1] per feature
        self.gate_proj = nn.Linear(model_dim, model_dim, bias=True)

        # Activations
        self.act = nn.ReLU6(inplace=False)
        self.gate_act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the gated residual block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, model_dim).

        Returns:
            torch.Tensor: Output tensor of same shape as input.
        """
        # Preserve residual
        residual = x

        # 1) Normalize input features
        x_norm = self.ln_in(x)  # (B, T, D)

        # 2) Expand -> Nonlinearity (ReLU6) -> Reduce
        hidden = self.fc_expand(x_norm)            # (B, T, H)
        hidden = self.act(hidden)                  # (B, T, H)
        projected = self.fc_reduce(hidden)         # (B, T, D)

        # 3) Compute gate from normalized residual
        gate_in = self.ln_gate(residual)           # (B, T, D)
        gate_logits = self.gate_proj(gate_in)      # (B, T, D)
        gate = self.gate_act(gate_logits)          # (B, T, D) values in (0,1)

        # 4) Gated residual combination
        combined = residual + gate * projected     # (B, T, D)

        # 5) Final normalization for stable outputs
        out = self.ln_out(combined)                # (B, T, D)

        return out

def get_inputs():
    """
    Generates a random input tensor matching the configured shapes.

    Returns:
        list: Single-element list containing the input tensor of shape (batch_size, seq_len, model_dim).
    """
    x = torch.randn(batch_size, seq_len, model_dim)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters required to instantiate the Model.

    Returns:
        list: [model_dim, hidden_dim]
    """
    return [model_dim, hidden_dim]