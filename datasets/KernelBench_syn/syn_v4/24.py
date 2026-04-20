import torch
import torch.nn as nn

# Configuration variables
BATCH_SIZE = 4
CHANNELS = 16
HEIGHT = 128
WIDTH = 128
PAD = 2               # replication padding on each side
POOL_KERNEL = 3
POOL_STRIDE = 2
RNN_HIDDEN = 32
RNN_NUM_LAYERS = 1

class Model(nn.Module):
    """
    Complex module combining replication padding, Lp pooling and an Elman RNN applied over spatial locations.

    Pipeline:
      1. Replication padding (ReplicationPad2d)
      2. Lp pooling (LPPool2d) to reduce spatial resolution
      3. Reshape pooled spatial grid into a sequence of feature vectors (seq_len = H_out * W_out)
      4. Process the sequence with a single-layer RNN (nn.RNN)
      5. Project RNN outputs back to channel dimension via a linear layer
      6. Reshape back to spatial grid and add a residual connection from the pooled features

    Input:
      x: Tensor of shape (N, C, H, W)

    Output:
      Tensor of shape (N, C, H_out, W_out) where H_out and W_out are the spatial dims after pooling.
    """
    def __init__(self):
        super(Model, self).__init__()
        # Replicate boundary values to pad the image
        self.pad = nn.ReplicationPad2d(PAD)
        # Lp pooling (p=2 is root mean square-like pooling)
        self.pool = nn.LPPool2d(norm_type=2, kernel_size=POOL_KERNEL, stride=POOL_STRIDE)
        # RNN processes a sequence of length H_out*W_out where each timestep has size CHANNELS
        self.rnn = nn.RNN(input_size=CHANNELS, hidden_size=RNN_HIDDEN, num_layers=RNN_NUM_LAYERS, nonlinearity='tanh', batch_first=False)
        # Linear projection from RNN hidden state back to CHANNELS for reconstruction
        self.fc = nn.Linear(RNN_HIDDEN, CHANNELS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the composed operations.

        Args:
            x (torch.Tensor): Input images of shape (N, C, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (N, C, H_out, W_out)
        """
        # 1) Replication padding
        x_padded = self.pad(x)  # (N, C, H+2*PAD, W+2*PAD)

        # 2) Lp pooling to reduce spatial resolution
        pooled = self.pool(x_padded)  # (N, C, H_out, W_out)
        N, C, H_out, W_out = pooled.shape

        # 3) Reshape pooled grid into a sequence of feature vectors
        #    from (N, C, H_out, W_out) -> (seq_len, N, C)
        seq_len = H_out * W_out
        seq = pooled.view(N, C, seq_len).permute(2, 0, 1)  # (seq_len, N, C)

        # 4) Process with RNN -> outputs (seq_len, N, RNN_HIDDEN)
        rnn_out, _ = self.rnn(seq)  # rnn_out: (seq_len, N, RNN_HIDDEN)

        # 5) Project each timestep's hidden state back to CHANNELS
        #    Flatten to apply linear then restore
        rnn_flat = rnn_out.contiguous().view(-1, RNN_HIDDEN)  # (seq_len * N, RNN_HIDDEN)
        projected = self.fc(rnn_flat)  # (seq_len * N, CHANNELS)
        projected = projected.view(seq_len, N, C).permute(1, 2, 0)  # (N, C, seq_len)

        # 6) Reshape back to spatial grid and add residual connection from pooled features
        out = projected.view(N, C, H_out, W_out) + pooled  # residual add

        return out

def get_inputs():
    """
    Returns a list with a single input tensor shaped (BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)
    """
    x = torch.randn(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)
    return [x]

def get_init_inputs():
    """
    No special initialization inputs required; return empty list to match expected interface.
    """
    return []