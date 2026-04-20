import torch
import torch.nn as nn

# Configuration
batch_size = 8
seq_len = 16
input_size = 32
hidden_size = 64
num_layers = 2
bidirectional = True
fc_hidden = 128
output_dim = 10
dropout = 0.1

class Model(nn.Module):
    """
    Sequence model that combines an LSTM encoder with attention-like pooling
    and gating using Hardswish and Hardsigmoid nonlinearities.

    Pipeline:
      1. LSTM over input sequence (batch_first=True).
      2. Elementwise Hardswish on the LSTM outputs.
      3. Learnable attention logits per timestep -> softmax over time.
      4. Attention-weighted pooling to create a context vector.
      5. Hardsigmoid gating (via a linear projection) applied to the context.
      6. Two-layer MLP with Hardswish nonlinearity to produce final logits.
      7. Log-Softmax over output classes.
    """
    def __init__(
        self,
        input_size: int = input_size,
        hidden_size: int = hidden_size,
        num_layers: int = num_layers,
        bidirectional: bool = bidirectional,
        fc_hidden: int = fc_hidden,
        output_dim: int = output_dim,
        dropout: float = dropout,
    ):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.encoder_dim = hidden_size * self.num_directions

        # Core recurrent encoder
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Nonlinearities
        self.hardswish = nn.Hardswish()
        self.hardsigmoid = nn.Hardsigmoid()

        # Attention projection (per time-step -> scalar)
        self.attn_proj = nn.Linear(self.encoder_dim, 1)

        # Gating projection to compute element-wise gate for the pooled context
        self.fc_gate = nn.Linear(self.encoder_dim, self.encoder_dim)

        # MLP head
        self.fc1 = nn.Linear(self.encoder_dim, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, input_size).

        Returns:
            torch.Tensor: Log-probabilities over output_dim classes, shape (batch, output_dim).
        """
        # LSTM encoding
        # output: (batch, seq_len, hidden_size * num_directions)
        output, (h_n, c_n) = self.lstm(x)

        # Nonlinear transform on timewise outputs
        output_hs = self.hardswish(output)  # (batch, seq_len, encoder_dim)

        # Attention logits per timestep
        attn_logits = self.attn_proj(output_hs).squeeze(-1)  # (batch, seq_len)
        attn_weights = torch.softmax(attn_logits, dim=1)     # (batch, seq_len)

        # Attention-weighted pooling to form context vector
        # context: (batch, encoder_dim)
        context = torch.einsum('bs,bsh->bh', attn_weights, output_hs)

        # Gating: produce element-wise gate and apply it
        gate = self.hardsigmoid(self.fc_gate(context))  # (batch, encoder_dim)
        gated_context = context * gate                  # (batch, encoder_dim)

        # MLP head with Hardswish nonlinearity
        hidden = self.hardswish(self.fc1(gated_context))  # (batch, fc_hidden)
        logits = self.fc2(hidden)                         # (batch, output_dim)

        # Return log-probabilities
        return torch.log_softmax(logits, dim=1)


def get_inputs():
    """
    Returns example input tensors for the model forward pass.

    Output:
      - x: Tensor of shape (batch_size, seq_len, input_size)
    """
    x = torch.randn(batch_size, seq_len, input_size)
    return [x]

def get_init_inputs():
    """
    Returns the arguments used to initialize the Model class with the module-level configuration.
    """
    return [input_size, hidden_size, num_layers, bidirectional, fc_hidden, output_dim, dropout]