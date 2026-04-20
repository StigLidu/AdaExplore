import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Sequence processor that combines an RNNCell with SyncBatchNorm and Hardtanh activation.
    At each time step the input is projected to the hidden dimension, processed by an RNNCell,
    merged with the projected input as a residual, normalized with SyncBatchNorm, and clipped
    by Hardtanh. The per-step activations are collected and a temporal max is returned.
    """
    def __init__(self, input_size: int, hidden_size: int, seq_len: int, use_relu: bool = False):
        """
        Initializes the model.

        Args:
            input_size (int): Size of features per time step.
            hidden_size (int): Size of RNN hidden state.
            seq_len (int): Expected sequence length (used for validation in forward).
            use_relu (bool): If True use ReLU nonlinearity for the RNNCell; otherwise tanh.
        """
        super(Model, self).__init__()
        nonlinearity = 'relu' if use_relu else 'tanh'
        self.rnn_cell = nn.RNNCell(input_size=input_size, hidden_size=hidden_size, nonlinearity=nonlinearity)
        # Project input into hidden space for a lightweight residual connection
        self.input_proj = nn.Linear(input_size, hidden_size, bias=False)
        # SyncBatchNorm works like BatchNorm but synchronized across processes;
        # it accepts inputs shaped (N, C, *) so (N, C) is valid here for hidden vectors.
        self.sync_bn = nn.SyncBatchNorm(num_features=hidden_size)
        # Hardtanh to clip activations into a bounded range, adding non-linearity
        self.hardtanh = nn.Hardtanh(min_val=-2.0, max_val=2.0)
        self.hidden_size = hidden_size
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes an input sequence.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            torch.Tensor: Tensor of shape (batch_size, hidden_size) containing the temporal max
                          over the processed per-time-step hidden activations.
        """
        if x.dim() != 3:
            raise ValueError(f"Expected input x to be 3D (batch, seq_len, input_size), got shape {x.shape}")
        batch_size, seq_len, input_size = x.shape
        if seq_len != self.seq_len:
            # Allow different seq_len at runtime but warn / adapt (keeping model flexible)
            # Update internal seq_len for this forward pass
            # (no layers depend on seq_len except for potential validations)
            self.seq_len = seq_len

        device = x.device
        dtype = x.dtype
        h = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch_size, input_size)
            proj = self.input_proj(x_t)  # (batch_size, hidden_size)
            h = self.rnn_cell(x_t, h)    # (batch_size, hidden_size)
            # Add a scaled residual from the projected input to encourage information flow
            h = h + 0.5 * proj
            # SyncBatchNorm expects channel dimension at dim=1 for (N, C) shaped tensors
            h_bn = self.sync_bn(h)
            h_act = self.hardtanh(h_bn)
            outputs.append(h_act.unsqueeze(1))  # (batch_size, 1, hidden_size)

        outputs_cat = torch.cat(outputs, dim=1)  # (batch_size, seq_len, hidden_size)
        # Temporal max pooling across the sequence length -> (batch_size, hidden_size)
        out_max = outputs_cat.max(dim=1)[0]
        return out_max

# Configuration for input generation / initialization
batch_size = 8
seq_len = 12
input_size = 64
hidden_size = 128
use_relu = True  # toggle RNN nonlinearity between 'relu' and 'tanh'

def get_inputs():
    """
    Returns example runtime inputs for the model: a random sequence tensor.
    Shape: (batch_size, seq_len, input_size)
    """
    x = torch.randn(batch_size, seq_len, input_size)
    return [x]

def get_init_inputs():
    """
    Returns the initialization arguments for the Model constructor in order:
    [input_size, hidden_size, seq_len, use_relu]
    """
    return [input_size, hidden_size, seq_len, use_relu]