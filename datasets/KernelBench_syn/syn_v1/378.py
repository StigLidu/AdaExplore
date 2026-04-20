import torch
import torch.nn as nn
import torch.nn.functional as F

# Configuration / default initialization parameters
batch_size = 8
seq_len = 64
input_size = 128
hidden_size = 256
num_layers = 2
bidirectional = True
out_dim = 192
dropout_prob = 0.1  # used inside the model for regularization

class Model(nn.Module):
    """
    Sequence encoder that:
      - Runs a multi-layer (optionally bidirectional) LSTM over the input sequence
      - Applies a randomized Leaky ReLU (RReLU) nonlinearity to the LSTM outputs
      - Computes an attention-weighted context vector from the activated LSTM outputs
      - Projects both a pooled representation and the context to a shared output dimension
      - Uses a Hardsigmoid gate to combine the two projected representations into the final output

    Inputs (constructed by get_inputs):
      x: Tensor of shape (batch, seq_len, input_size)

    Initialization inputs (constructed by get_init_inputs):
      input_size, hidden_size, num_layers, bidirectional, out_dim
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 bidirectional: bool = False,
                 out_dim: int = 128):
        super(Model, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.out_dim = out_dim

        self.num_directions = 2 if bidirectional else 1

        # Core recurrent encoder
        # Use batch_first=True so inputs are (batch, seq_len, input_size)
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional,
                            dropout=dropout_prob if num_layers > 1 else 0.0)

        # Non-linearities
        # RReLU applies element-wise randomized leaky ReLU during training
        self.rrelu = nn.RReLU(lower=0.125, upper=0.333)
        # Hardsigmoid used as a gating function
        self.hardsigmoid = nn.Hardsigmoid()

        # Projections: project pooled/contextual representations to out_dim
        self.pool_proj = nn.Linear(hidden_size * self.num_directions, out_dim)
        self.ctx_proj = nn.Linear(hidden_size * self.num_directions, out_dim)

        # Small attention mechanism: score each time step to form context
        self.attn_score = nn.Linear(hidden_size * self.num_directions, 1, bias=False)

        # Optional final projection after gating (keeps dimensionality stable)
        self.final_linear = nn.Linear(out_dim, out_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining LSTM, RReLU, attention pooling, gating with Hardsigmoid,
        and final projection.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, input_size)

        Returns:
            torch.Tensor: Output tensor of shape (batch, out_dim)
        """
        # Encode sequence with LSTM -> (batch, seq_len, hidden * num_directions)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Apply randomized leaky ReLU element-wise
        # (keeps same shape as lstm_out)
        activated = self.rrelu(lstm_out)

        # Compute a simple mean-pooled representation across the time dimension
        pooled = torch.mean(activated, dim=1)  # shape: (batch, hidden * num_directions)

        # Project pooled representation to out_dim
        pooled_proj = self.pool_proj(pooled)  # (batch, out_dim)

        # Attention scores for each time step -> (batch, seq_len, 1) -> squeeze -> (batch, seq_len)
        attn_logits = self.attn_score(activated).squeeze(-1)

        # Normalize attention over time dimension
        attn_weights = torch.softmax(attn_logits, dim=1)  # (batch, seq_len)

        # Compute attention-weighted context vector: sum over time of weight * activated
        # use einsum for clarity: attn_weights[b, s] * activated[b, s, d] -> context[b, d]
        context = torch.einsum("bs,bsd->bd", attn_weights, activated)  # (batch, hidden * num_directions)

        # Project context into the same output space
        context_proj = self.ctx_proj(context)  # (batch, out_dim)

        # Compute gate from context projection and transform via Hardsigmoid
        gate_pre = self.dropout(context_proj)  # small dropout before gating
        gate = self.hardsigmoid(gate_pre)  # values in [0,1], shape (batch, out_dim)

        # Combine pooled_proj and context_proj using the gate:
        # output = gate * pooled_proj + (1 - gate) * context_proj
        combined = gate * pooled_proj + (1.0 - gate) * context_proj

        # Final linear + nonlinearity to produce stable output
        out = torch.tanh(self.final_linear(combined))

        return out

def get_inputs():
    """
    Returns a list containing a single input tensor x of shape:
      (batch_size, seq_len, input_size)
    """
    x = torch.randn(batch_size, seq_len, input_size)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for constructing the Model:
      [input_size, hidden_size, num_layers, bidirectional, out_dim]
    """
    return [input_size, hidden_size, num_layers, bidirectional, out_dim]