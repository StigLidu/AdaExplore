import torch
import torch.nn as nn

# Configuration
BATCH = 8
SEQ_LEN = 128
EMBED = 512
HIDDEN = 2048
DROPOUT_P = 0.1
LOGSOFTMAX_DIM = -1

class Model(nn.Module):
    """
    Complex model that projects sequence embeddings into a higher-dimensional
    hidden space, applies normalization and FeatureAlphaDropout, projects back
    and produces a log-probability distribution across the embedding dimension.

    Computation pattern (high-level):
      1. Project input x: (B, S, E) -> (B, S, H) via linear projection (W1, b1)
      2. LayerNorm over hidden dimension
      3. FeatureAlphaDropout over features/channels
      4. Project back to embedding dim: (B, S, H) -> (B, S, E) via W2, b2
      5. Residual addition with original input and second LayerNorm
      6. LogSoftmax across embedding dimension to get log-probabilities
    """
    def __init__(self):
        super(Model, self).__init__()
        # Weight matrices for two linear projections implemented as parameters
        self.W1 = nn.Parameter(torch.randn(EMBED, HIDDEN) * (1.0 / EMBED**0.5))
        self.b1 = nn.Parameter(torch.zeros(HIDDEN))
        self.W2 = nn.Parameter(torch.randn(HIDDEN, EMBED) * (1.0 / HIDDEN**0.5))
        self.b2 = nn.Parameter(torch.zeros(EMBED))

        # Normalization layers
        self.ln_hidden = nn.LayerNorm(HIDDEN)
        self.ln_out = nn.LayerNorm(EMBED)

        # Dropout that is specialized for feature/channel style dropout
        self.feature_alpha_dropout = nn.FeatureAlphaDropout(p=DROPOUT_P)

        # Final log-softmax to produce log-probabilities along embedding dimension
        self.log_softmax = nn.LogSoftmax(dim=LOGSOFTMAX_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (BATCH, SEQ_LEN, EMBED)

        Returns:
            Tensor of shape (BATCH, SEQ_LEN, EMBED) containing log-probabilities
            over the embedding dimension for each sequence position.
        """
        # 1) Linear projection to hidden space
        # x: (B, S, E), W1: (E, H) -> hidden: (B, S, H)
        hidden = torch.matmul(x, self.W1) + self.b1

        # 2) Layer normalization across hidden features
        hidden = self.ln_hidden(hidden)

        # 3) FeatureAlphaDropout (randomly masks whole features/channels)
        # Note: dropout is active during training; keep this module compatible
        # with eval() vs train() semantics.
        hidden = self.feature_alpha_dropout(hidden)

        # 4) Project back to embedding dimension
        out = torch.matmul(hidden, self.W2) + self.b2  # (B, S, E)

        # 5) Residual connection + final LayerNorm
        out = self.ln_out(out + x)

        # 6) Convert to log-probabilities across embedding dimension
        out = self.log_softmax(out)

        return out

def get_inputs():
    """
    Returns:
        List containing one input tensor with shape (BATCH, SEQ_LEN, EMBED)
    """
    x = torch.randn(BATCH, SEQ_LEN, EMBED)
    return [x]

def get_init_inputs():
    """
    No special external initialization inputs required; weights are internal parameters.
    """
    return []