import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex sequence module that processes an input sequence with a multi-layer GRU,
    applies SiLU activation and FeatureAlphaDropout, performs pooled projection and
    gating, and combines with the last input timestep as a residual.

    Forward signature:
        x: Tensor of shape (batch, seq_len, input_size)
        h0: Optional initial hidden state tensor of shape (num_layers * num_directions, batch, hidden_size)

    Returns:
        Tensor of shape (batch, input_size) -- combined output per batch element
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        bidirectional: bool = True,
        dropout_p: float = 0.0,
        fa_dropout_p: float = 0.0,
    ):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Recurrent backbone
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_p if num_layers > 1 else 0.0,
        )

        # Non-linearity and channel dropout
        self.silu = nn.SiLU()
        self.feature_dropout = nn.FeatureAlphaDropout(p=fa_dropout_p)

        # Projections that map pooled sequence features back to input feature dimension
        hidden_proj_dim = hidden_size * self.num_directions
        self.pool_proj = nn.Linear(hidden_proj_dim, input_size)
        self.last_proj = nn.Linear(hidden_proj_dim, input_size)

        # Small learnable scale to modulate residual blending
        self.residual_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        """
        Process sequence and produce a single feature vector per batch element.

        Steps:
            1. Run a GRU over the sequence.
            2. Apply SiLU to the GRU outputs.
            3. Apply FeatureAlphaDropout across channels.
            4. Pool (mean) across the time dimension to get a global descriptor.
            5. Project pooled descriptor and the last timestep descriptor back to input_size.
            6. Compute a sigmoid gate from the pooled projection and gate the last projection.
            7. Add a scaled residual from the last input timestep and return.

        Args:
            x: (batch, seq_len, input_size)
            h0: (num_layers * num_directions, batch, hidden_size)

        Returns:
            out: (batch, input_size)
        """
        # 1. GRU
        outputs, h_n = self.gru(x, h0)  # outputs: (batch, seq_len, hidden_proj_dim)

        # 2. Non-linearity
        activated = self.silu(outputs)

        # 3. Channel dropout
        dropped = self.feature_dropout(activated)

        # 4. Pool across time to get global context
        pooled = dropped.mean(dim=1)  # (batch, hidden_proj_dim)

        # 5. Projections back to input feature dimension
        pooled_proj = self.pool_proj(pooled)      # (batch, input_size)
        last_timestep = dropped[:, -1, :]         # (batch, hidden_proj_dim)
        last_proj = self.last_proj(last_timestep) # (batch, input_size)

        # 6. Gate and combine
        gate = torch.sigmoid(pooled_proj)         # (batch, input_size)
        gated = gate * last_proj                  # (batch, input_size)

        # 7. Residual from last input timestep (align dims)
        last_input = x[:, -1, :]                  # (batch, input_size)
        out = last_input * self.residual_scale + gated

        return out

# Module-level configuration
BATCH_SIZE = 8
SEQ_LEN = 50
INPUT_SIZE = 64
HIDDEN_SIZE = 128
NUM_LAYERS = 2
BIDIRECTIONAL = True
GRU_DROPOUT = 0.1
FEATURE_DROP_P = 0.2

def get_inputs():
    """
    Returns:
        x: (batch, seq_len, input_size)
        h0: (num_layers * num_directions, batch, hidden_size)
    """
    num_directions = 2 if BIDIRECTIONAL else 1
    x = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_SIZE)
    h0 = torch.randn(NUM_LAYERS * num_directions, BATCH_SIZE, HIDDEN_SIZE)
    return [x, h0]

def get_init_inputs():
    """
    Returns initialization parameters to construct the Model:
        [input_size, hidden_size, num_layers, bidirectional, dropout_p, fa_dropout_p]
    """
    return [INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, BIDIRECTIONAL, GRU_DROPOUT, FEATURE_DROP_P]