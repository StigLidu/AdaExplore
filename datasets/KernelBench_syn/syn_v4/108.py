import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Attention-style sequence pooling model that demonstrates a combination of:
    - Linear projections
    - Non-linear activations (ReLU6)
    - Learnable gating (Hardsigmoid)
    - Sequence scoring with LogSoftmax
    - Weighted aggregation (einsum)
    - Final classification with another LogSoftmax

    Input: tensor of shape (batch_size, seq_len, input_dim)
    Output: log-probabilities over out_dim classes (tensor of shape (batch_size, out_dim))
    """
    def __init__(self, input_dim: int, hidden_dim: int, proj_dim: int, out_dim: int):
        super(Model, self).__init__()
        # First projection from input_dim -> hidden_dim
        self.lin1 = nn.Linear(input_dim, hidden_dim, bias=True)
        # Bounded ReLU activation
        self.relu6 = nn.ReLU6()
        # Second projection from hidden_dim -> proj_dim (feature space we pool over)
        self.lin2 = nn.Linear(hidden_dim, proj_dim, bias=True)
        # Gating nonlinearity that squashes to [0,1]
        self.hardsig = nn.Hardsigmoid()
        # Scalar score per sequence element (used to compute attention weights)
        self.score_linear = nn.Linear(proj_dim, 1, bias=False)
        # Normalize sequence scores into log-probabilities across seq_len
        self.logsoftmax_seq = nn.LogSoftmax(dim=1)
        # Final classifier from pooled representation -> out_dim
        self.out_linear = nn.Linear(proj_dim, out_dim, bias=True)
        # Final layer normalization and scaling
        self.layernorm = nn.LayerNorm(out_dim)
        self.scale = nn.Parameter(torch.tensor(0.75))  # learnable scaling of logits
        # Output log-softmax
        self.logsoftmax_out = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            Tensor of shape (batch_size, out_dim) with log-probabilities.
        """
        # Project input into hidden space
        # shape -> (batch, seq_len, hidden_dim)
        h = self.lin1(x)

        # Non-linearity
        h = self.relu6(h)

        # Project into pooling feature space
        # shape -> (batch, seq_len, proj_dim)
        p = self.lin2(h)

        # Compute a gate in [0,1] and apply element-wise to projection
        gate = self.hardsig(p)
        gated = p * gate  # elementwise gating

        # Compute unnormalized scores per sequence element
        # shape -> (batch, seq_len, 1) -> squeeze -> (batch, seq_len)
        scores = self.score_linear(gated).squeeze(-1)

        # Convert to log-probabilities across the sequence dimension
        log_weights = self.logsoftmax_seq(scores)

        # Convert log-weights to probabilities and perform weighted sum across sequence
        weights = torch.exp(log_weights)  # shape -> (batch, seq_len)
        # Weighted aggregate of gated features -> (batch, proj_dim)
        pooled = torch.einsum('bse,bs->be', gated, weights)

        # Final classification head
        logits = self.out_linear(pooled)           # (batch, out_dim)
        scaled = logits * self.scale               # learned scalar scaling
        normed = self.layernorm(scaled)            # normalize logits
        out_log_probs = self.logsoftmax_out(normed)  # final log-probabilities

        return out_log_probs

# Configuration for input generation
batch_size = 8
seq_len = 128
input_dim = 256
hidden_dim = 512
proj_dim = 128
out_dim = 10

def get_inputs():
    """
    Returns a list containing a single input tensor of shape:
    (batch_size, seq_len, input_dim)
    """
    x = torch.randn(batch_size, seq_len, input_dim)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters required to construct the model.
    """
    return [input_dim, hidden_dim, proj_dim, out_dim]