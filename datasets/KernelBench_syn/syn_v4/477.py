import torch
import torch.nn as nn

"""
Complex sequence processing model that combines MultiheadAttention, LSTMCell, and LogSigmoid.
Structure follows the provided examples: Model class, get_inputs, get_init_inputs, and module-level configs.
"""

# Configuration variables
batch_size = 8
seq_len = 64
embed_dim = 256  # must be divisible by num_heads
num_heads = 8
lstm_hidden = 512

class Model(nn.Module):
    """
    Sequence model that:
      - Applies self-attention over the input sequence (nn.MultiheadAttention)
      - Concatenates attention outputs with the original inputs
      - Processes the concatenated sequence step-by-step through an LSTMCell
      - Pools the LSTM hidden states and projects back to embedding dimension
      - Applies LogSigmoid nonlinearity to the final projection
    """
    def __init__(self, embed_dim: int, num_heads: int, lstm_hidden: int):
        """
        Initializes the model components.

        Args:
            embed_dim (int): Dimension of the input embeddings.
            num_heads (int): Number of attention heads for MultiheadAttention.
            lstm_hidden (int): Hidden size for the LSTMCell.
        """
        super(Model, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.lstm_hidden = lstm_hidden

        # Multi-head self-attention (expects input shape (S, N, E) by default)
        self.mha = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, batch_first=False)

        # LSTMCell processes concatenated [input; attn_output] at each time step
        self.lstm_cell = nn.LSTMCell(input_size=self.embed_dim * 2, hidden_size=self.lstm_hidden)

        # Final projection from LSTM hidden dim back to embedding dim
        self.out_proj = nn.Linear(self.lstm_hidden, self.embed_dim)

        # LogSigmoid non-linearity applied to the final projected outputs
        self.logsigmoid = nn.LogSigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input sequence tensor of shape (seq_len, batch_size, embed_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, embed_dim) after pooling, projection, and LogSigmoid.
        """
        # Self-attention over the sequence
        # attn_output: (S, N, E)
        attn_output, _ = self.mha(x, x, x)

        # Concatenate original input and attention output along feature dimension -> (S, N, 2E)
        combined = torch.cat([x, attn_output], dim=2)

        S, N, _ = combined.shape

        # Initialize LSTMCell hidden and cell states (N, H)
        h = torch.zeros(N, self.lstm_hidden, device=x.device, dtype=x.dtype)
        c = torch.zeros(N, self.lstm_hidden, device=x.device, dtype=x.dtype)

        outputs = []
        # Process sequence step-by-step through LSTMCell
        for t in range(S):
            input_t = combined[t]  # (N, 2E)
            h, c = self.lstm_cell(input_t, (h, c))  # each is (N, H)
            outputs.append(h)

        # Stack to (S, N, H)
        outputs = torch.stack(outputs, dim=0)

        # Temporal pooling: mean over time -> (N, H)
        pooled = outputs.mean(dim=0)

        # Project back to embedding dimension -> (N, E)
        proj = self.out_proj(pooled)

        # Apply LogSigmoid non-linearity elementwise -> (N, E)
        out = self.logsigmoid(proj)

        return out

def get_inputs():
    """
    Returns a list containing a single input tensor shaped (seq_len, batch_size, embed_dim).
    """
    x = torch.randn(seq_len, batch_size, embed_dim)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for the Model: [embed_dim, num_heads, lstm_hidden].
    """
    return [embed_dim, num_heads, lstm_hidden]