import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Sequence processor that stacks two GRUCells, applies synchronized batch normalization
    over the temporal hidden-state outputs, clamps activations with Hardtanh, and projects
    to an output space with a linear layer.

    Computation pattern:
      - For each timestep t:
          h1_t = GRUCell1(x_t, h1_{t-1})
          h2_t = GRUCell2(h1_t, h2_{t-1})
      - Stack h2_t across time -> (batch, seq_len, hidden_size)
      - Apply SyncBatchNorm across the feature dimension (after flattening batch*time)
      - Apply Hardtanh activation
      - Temporal average pooling -> (batch, hidden_size)
      - Linear projection -> (batch, output_dim)

    This pattern mixes recurrent cells, normalization across batch+time, nonlinearity,
    and a final linear mapping to produce per-sequence outputs.
    """
    def __init__(self, input_size: int, hidden_size: int, output_dim: int):
        """
        Args:
            input_size: Dimensionality of input features per timestep.
            hidden_size: Hidden state dimensionality for GRU cells.
            output_dim: Dimensionality of final output per sequence.
        """
        super(Model, self).__init__()
        # Two-layer GRUCell stack (layerwise recurrence implemented manually)
        self.gru1 = nn.GRUCell(input_size, hidden_size)
        self.gru2 = nn.GRUCell(hidden_size, hidden_size)
        # SyncBatchNorm operates on feature channels (hidden_size). Using affine params.
        self.bn = nn.SyncBatchNorm(hidden_size, affine=True)
        # Hardtanh clamps values to a bounded range; helps control activation magnitudes
        self.act = nn.Hardtanh(min_val=-2.0, max_val=2.0)
        # Final projection from hidden state summary to desired output dimension
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes an input sequence and returns a per-sequence output.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            Tensor of shape (batch_size, output_dim)
        """
        if x.dim() != 3:
            raise ValueError("Expected input x to have 3 dimensions (batch, seq_len, input_size)")

        batch_size, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype

        # Initialize hidden states for both GRU layers to zeros
        h1 = torch.zeros(batch_size, self.gru1.hidden_size, device=device, dtype=dtype)
        h2 = torch.zeros(batch_size, self.gru2.hidden_size, device=device, dtype=dtype)

        outputs = []
        # Recurrently process the sequence timestep by timestep
        for t in range(seq_len):
            xt = x[:, t, :]            # (batch_size, input_size)
            h1 = self.gru1(xt, h1)     # (batch_size, hidden_size)
            h2 = self.gru2(h1, h2)     # (batch_size, hidden_size)
            outputs.append(h2)

        # Stack temporal outputs: (batch_size, seq_len, hidden_size)
        seq_hidden = torch.stack(outputs, dim=1)

        # Prepare for batch-norm: flatten batch and time into a single dimension for per-feature normalization
        seq_flat = seq_hidden.contiguous().view(batch_size * seq_len, -1)  # (batch*seq_len, hidden_size)
        # SyncBatchNorm expects (N, C, ...) with C as num_features; given (N, C) it's fine.
        seq_bn = self.bn(seq_flat)  # (batch*seq_len, hidden_size)

        # Nonlinearity and reshape back to (batch, seq_len, hidden_size)
        seq_act = self.act(seq_bn).view(batch_size, seq_len, -1)

        # Temporal average pooling to get a sequence-level representation
        seq_summary = seq_act.mean(dim=1)  # (batch_size, hidden_size)

        # Final linear projection
        out = self.fc(seq_summary)  # (batch_size, output_dim)
        return out

# Configuration / hyperparameters
BATCH_SIZE = 8
SEQ_LEN = 16
INPUT_SIZE = 512
HIDDEN_SIZE = 1024
OUTPUT_DIM = 256

def get_inputs():
    """
    Returns:
        List containing the input sequence tensor:
          - x: torch.Tensor of shape (BATCH_SIZE, SEQ_LEN, INPUT_SIZE)
    """
    x = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_SIZE)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for constructing the Model:
        [input_size, hidden_size, output_dim]
    """
    return [INPUT_SIZE, HIDDEN_SIZE, OUTPUT_DIM]