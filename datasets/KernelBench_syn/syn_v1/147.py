import torch
import torch.nn as nn
from typing import Optional, List

class Model(nn.Module):
    """
    Complex sequence classification model that:
    - Projects input features
    - Runs a (possibly bidirectional) GRU over the sequence
    - Applies SyncBatchNorm across the feature/channel dimension for the temporal outputs
    - Pools the normalized sequence outputs
    - Adds a residual connection from the projected input
    - Projects to class logits and returns log-probabilities via LogSoftmax
    """
    def __init__(
        self,
        hidden_size: int = 256,
        num_layers: int = 2,
        input_dim: int = 128,
        num_classes: int = 40,
        bidirectional: bool = True,
    ):
        """
        Args:
            hidden_size: Hidden size per GRU direction.
            num_layers: Number of stacked GRU layers.
            input_dim: Dimensionality of input features.
            num_classes: Number of output classes for classification.
            bidirectional: Whether to use a bidirectional GRU.
        """
        super(Model, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1

        # Project raw input features up to the GRU feature dimension (match GRU output channels)
        self.input_proj = nn.Linear(self.input_dim, self.hidden_size * self.num_directions)

        # GRU: takes the projected features as input
        self.gru = nn.GRU(
            input_size=self.hidden_size * self.num_directions,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

        # SyncBatchNorm over the channel dimension (we will reshape to N, C, L)
        self.norm = nn.SyncBatchNorm(self.hidden_size * self.num_directions)

        # Classifier projects pooled representation to logits
        self.classifier = nn.Linear(self.hidden_size * self.num_directions, self.num_classes)

        # LogSoftmax for stable log-probabilities over classes
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim).
            h0: Optional initial hidden state for the GRU with shape
                (num_layers * num_directions, batch, hidden_size). If None, GRU defaults to zeros.

        Returns:
            Log-probabilities over classes: tensor of shape (batch, num_classes).
        """
        # Project input features
        # x_proj shape: (batch, seq_len, feat) where feat = hidden_size * num_directions
        x_proj = self.input_proj(x)

        # Run GRU: output shape (batch, seq_len, hidden_size * num_directions)
        gru_out, h_n = self.gru(x_proj, h0)

        # SyncBatchNorm expects channels in dim=1 for (N, C, L,...). Permute accordingly.
        # From (batch, seq_len, channels) -> (batch, channels, seq_len)
        y = gru_out.permute(0, 2, 1)

        # Normalize across batch & sequence for each channel
        y = self.norm(y)

        # Back to (batch, seq_len, channels)
        y = y.permute(0, 2, 1)

        # Temporal pooling: mean over sequence length
        pooled = y.mean(dim=1)  # (batch, channels)

        # Residual: also pool the projected input and add (same channel dim)
        input_residual = x_proj.mean(dim=1)
        combined = pooled + input_residual

        # Non-linear gating (simple tanh)
        gated = torch.tanh(combined)

        # Classifier + LogSoftmax
        logits = self.classifier(gated)
        log_probs = self.log_softmax(logits)

        return log_probs

# Configuration / default sizes
BATCH = 8
SEQ_LEN = 64
INPUT_DIM = 128
HIDDEN_SIZE = 256
NUM_LAYERS = 2
NUM_CLASSES = 40
BIDIRECTIONAL = True

def get_inputs() -> List[torch.Tensor]:
    """
    Returns runtime input tensors to feed into Model.forward:
    - x: (batch, seq_len, input_dim)
    - h0: initial hidden state for GRU with shape (num_layers * num_directions, batch, hidden_size)
    """
    x = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
    num_directions = 2 if BIDIRECTIONAL else 1
    h0 = torch.randn(NUM_LAYERS * num_directions, BATCH, HIDDEN_SIZE)
    return [x, h0]

def get_init_inputs() -> List:
    """
    Returns initialization parameters for the Model constructor in the same order:
    (hidden_size, num_layers, input_dim, num_classes, bidirectional)
    """
    return [HIDDEN_SIZE, NUM_LAYERS, INPUT_DIM, NUM_CLASSES, BIDIRECTIONAL]