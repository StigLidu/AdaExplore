import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Sequence-aware transformation block that combines linear projections with multiple
    nonlinearities and an attention-style weighting to produce a residual-enhanced output.

    Computation steps:
      1. Project input features to a hidden space (z1 = x @ W1 + b1).
      2. Apply CELU nonlinearity to hidden representation.
      3. Project hidden back to input dimension and apply Hardswish (z2 = Hardswish(a1 @ W2 + b2)).
      4. Compute attention logits from the hidden representation (att_logits = a1 @ W_att).
      5. Softmax across the sequence dimension to obtain attention weights and compute a context vector
         as the weighted sum of z2 across the sequence.
      6. Broadcast-add the context to z2 (residual) and apply LeakyReLU as a final nonlinearity.

    Input:
      x: Tensor of shape (batch_size, seq_len, dim)

    Output:
      Tensor of shape (batch_size, seq_len, dim)
    """
    def __init__(self, dim: int, hidden_dim: int, negative_slope: float = 0.1):
        """
        Initializes the parameters and nonlinear layers.

        Args:
            dim (int): Input feature dimensionality.
            hidden_dim (int): Hidden projection dimensionality.
            negative_slope (float): Negative slope for LeakyReLU.
        """
        super(Model, self).__init__()

        # Linear projection parameters as trainable parameters
        self.w1 = nn.Parameter(torch.empty(dim, hidden_dim))
        self.b1 = nn.Parameter(torch.empty(hidden_dim))
        self.w2 = nn.Parameter(torch.empty(hidden_dim, dim))
        self.b2 = nn.Parameter(torch.empty(dim))
        self.w_att = nn.Parameter(torch.empty(hidden_dim, 1))  # produce scalar logit per time-step

        # Nonlinear modules from the provided list
        self.celu = nn.CELU()           # element-wise CELU
        self.hardswish = nn.Hardswish() # element-wise Hardswish
        self.leakyrelu = nn.LeakyReLU(negative_slope=negative_slope)

        # Initialize parameters with standard practices
        nn.init.xavier_uniform_(self.w1)
        nn.init.zeros_(self.b1)
        nn.init.xavier_uniform_(self.w2)
        nn.init.zeros_(self.b2)
        nn.init.xavier_uniform_(self.w_att)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, dim).

        Returns:
            out: Output tensor of shape (batch_size, seq_len, dim).
        """
        # Project to hidden space: (batch, seq, hidden_dim)
        z1 = torch.matmul(x, self.w1) + self.b1

        # Nonlinear transform in hidden space
        a1 = self.celu(z1)

        # Project back to original dim and apply hardswish: (batch, seq, dim)
        z2 = torch.matmul(a1, self.w2) + self.b2
        z2 = self.hardswish(z2)

        # Attention logits from hidden representation: (batch, seq)
        att_logits = torch.matmul(a1, self.w_att).squeeze(-1)

        # Normalize across sequence dimension to get attention weights: (batch, seq, 1)
        att_weights = F.softmax(att_logits, dim=1).unsqueeze(-1)

        # Context vector: weighted sum over sequence of z2 -> (batch, dim)
        context = (att_weights * z2).sum(dim=1)

        # Broadcast-add context to every time-step to form a residual, then final activation
        out = z2 + context.unsqueeze(1)
        out = self.leakyrelu(out)

        return out

# Module-level configuration variables
batch_size = 32
seq_len = 128
dim = 512
hidden_dim = 256
negative_slope = 0.1

def get_inputs():
    """
    Returns a list containing the primary input tensor for the model:
      - x: shape (batch_size, seq_len, dim), dtype float32
    """
    x = torch.randn(batch_size, seq_len, dim, dtype=torch.float32)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for constructing the Model:
      [dim, hidden_dim, negative_slope]
    """
    return [dim, hidden_dim, negative_slope]