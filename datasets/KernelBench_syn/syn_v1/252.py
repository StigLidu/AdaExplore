import torch
import torch.nn as nn

"""
Complex PyTorch module combining:
- nn.MaxPool3d for spatial reduction of 3D volumes
- nn.RNNCell for temporal recurrence across a sequence of 3D volumes
- nn.SELU activation for self-normalizing non-linearity
Structure follows the example pattern:
- Model class (inherits nn.Module)
- get_inputs() to produce example inputs
- get_init_inputs() to produce constructor parameters
Module-level configuration variables are defined below.
"""

# Module-level configuration
BATCH_SIZE = 8
SEQ_LEN = 12
CHANNELS = 4
DEPTH = 16
HEIGHT = 24
WIDTH = 24

POOL_KERNEL = (2, 2, 2)  # MaxPool3d kernel
HIDDEN_SIZE = 128  # RNN hidden dimension

# Compute post-pooling spatial dimensions (integer division)
POOLED_DEPTH = DEPTH // POOL_KERNEL[0]
POOLED_HEIGHT = HEIGHT // POOL_KERNEL[1]
POOLED_WIDTH = WIDTH // POOL_KERNEL[2]

# Flattened input feature size after pooling
INPUT_FEATURE_SIZE = CHANNELS * POOLED_DEPTH * POOLED_HEIGHT * POOLED_WIDTH


class Model(nn.Module):
    """
    A model that processes a sequence of 3D volumes (e.g., volumetric frames).
    Per time-step pipeline:
      1. MaxPool3d to reduce spatial resolution
      2. SELU activation
      3. Flatten spatial dimensions to a vector per batch
      4. RNNCell to update hidden state
      5. Residual projection from input vector into hidden space added to RNN output
      6. SELU activation on the combined hidden state

    The forward method returns all hidden states stacked across the time dimension:
      Tensor shape -> (SEQ_LEN, BATCH_SIZE, HIDDEN_SIZE)
    """
    def __init__(self, input_size: int, hidden_size: int, pool_kernel=(2, 2, 2)):
        """
        Args:
            input_size (int): Flattened size after pooling (channels * d * h * w).
            hidden_size (int): Hidden size for the RNNCell.
            pool_kernel (tuple): Kernel size for MaxPool3d.
        """
        super(Model, self).__init__()
        self.pool = nn.MaxPool3d(kernel_size=pool_kernel, stride=pool_kernel)
        self.activation = nn.SELU()
        # RNNCell with tanh nonlinearity for recurrence
        self.rnn_cell = nn.RNNCell(input_size=input_size, hidden_size=hidden_size, nonlinearity='tanh')
        # Residual projection to map pooled input into hidden dimension before combining
        self.input_residual = nn.Linear(in_features=input_size, out_features=hidden_size)

    def forward(self, x: torch.Tensor, h0: torch.Tensor = None) -> torch.Tensor:
        """
        Processes a sequence of 3D volumes.

        Args:
            x (torch.Tensor): Input tensor of shape (SEQ_LEN, BATCH, C, D, H, W)
            h0 (torch.Tensor, optional): Initial hidden state of shape (BATCH, HIDDEN_SIZE).
                                         If None, initialized to zeros on the same device/dtype.

        Returns:
            torch.Tensor: Stacked hidden states of shape (SEQ_LEN, BATCH, HIDDEN_SIZE)
        """
        seq_len, batch, c, d, h, w = x.shape
        device = x.device
        dtype = x.dtype

        # Initialize hidden state if not provided
        if h0 is None:
            h = torch.zeros(batch, self.rnn_cell.hidden_size, device=device, dtype=dtype)
        else:
            h = h0

        hidden_states = []
        for t in range(seq_len):
            # Extract time-step volume: shape (BATCH, C, D, H, W)
            xt = x[t]
            # Spatial pooling: shape -> (BATCH, C, D', H', W')
            pooled = self.pool(xt)
            # Activation
            activated = self.activation(pooled)
            # Flatten spatial dims: (BATCH, input_size)
            flattened = activated.view(batch, -1)
            # RNNCell update: (BATCH, HIDDEN_SIZE)
            h_rnn = self.rnn_cell(flattened, h)
            # Residual projection from input into hidden space
            residual = self.input_residual(flattened)
            # Combine and apply SELU activation to produce next hidden state
            h = self.activation(h_rnn + residual)
            hidden_states.append(h)

        # Stack hidden states: (SEQ_LEN, BATCH, HIDDEN_SIZE)
        hidden_stack = torch.stack(hidden_states, dim=0)
        return hidden_stack


def get_inputs():
    """
    Generate a batch of random volumetric sequences.

    Returns:
        list: [x] where x is a tensor of shape (SEQ_LEN, BATCH_SIZE, CHANNELS, DEPTH, HEIGHT, WIDTH)
    """
    x = torch.randn(SEQ_LEN, BATCH_SIZE, CHANNELS, DEPTH, HEIGHT, WIDTH)
    return [x]


def get_init_inputs():
    """
    Returns constructor parameters for the Model: [input_size, hidden_size]

    Returns:
        list: [INPUT_FEATURE_SIZE, HIDDEN_SIZE]
    """
    return [INPUT_FEATURE_SIZE, HIDDEN_SIZE]