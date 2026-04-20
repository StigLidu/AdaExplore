import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex model that combines BatchNorm2d, LPPool1d and an RNN to process a sequence
    of feature maps. The model:
      - Applies BatchNorm2d over (batch, channels, seq_len, feat_dim)
      - Pools the feature dimension using LPPool1d for each time step and channel
      - Flattens pooled channels to produce a per-time-step feature vector
      - Feeds the sequence of vectors into an RNN (batch_first=True)
      - Aggregates RNN outputs by concatenating the time-wise mean and the last layer hidden state
      - Applies a final linear projection to produce outputs per batch
    """
    def __init__(self,
                 channels: int,
                 feat_dim: int,
                 pool_kernel: int,
                 p: int,
                 rnn_hidden: int,
                 rnn_layers: int,
                 linear_out: int):
        """
        Initializes the combined model.

        Args:
            channels (int): Number of channels in the input tensor.
            feat_dim (int): Size of the feature dimension (length to pool over).
            pool_kernel (int): Kernel size for LPPool1d (stride will be set equal to kernel for simplicity).
            p (int): The p-norm to use in LPPool1d.
            rnn_hidden (int): Hidden size for the RNN.
            rnn_layers (int): Number of RNN layers.
            linear_out (int): Output feature size of the final linear layer.
        """
        super(Model, self).__init__()

        # BatchNorm2d expects num_features == channels
        self.bn = nn.BatchNorm2d(num_features=channels)

        # Use stride = pool_kernel to simplify pooled length computation and avoid overlap
        self.pool = nn.LPPool1d(norm_type=p, kernel_size=pool_kernel, stride=pool_kernel)

        # Compute pooled feature length so we can size the RNN input
        if feat_dim < pool_kernel:
            raise ValueError("feat_dim must be >= pool_kernel to allow pooling.")
        pooled_len = (feat_dim - pool_kernel) // pool_kernel + 1
        rnn_input_size = channels * pooled_len

        # RNN to process the sequence of pooled feature vectors
        self.rnn = nn.RNN(input_size=rnn_input_size, hidden_size=rnn_hidden, num_layers=rnn_layers, batch_first=True, nonlinearity='tanh')

        # Final projection: we will concat time-mean and last hidden (both size rnn_hidden) -> 2 * rnn_hidden
        self.fc = nn.Linear(2 * rnn_hidden, linear_out)
        self.relu = nn.ReLU()

        # store some config for introspection / debugging
        self._config = {
            "channels": channels,
            "feat_dim": feat_dim,
            "pool_kernel": pool_kernel,
            "p": p,
            "rnn_hidden": rnn_hidden,
            "rnn_layers": rnn_layers,
            "linear_out": linear_out,
            "pooled_len": pooled_len,
            "rnn_input_size": rnn_input_size
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, seq_len, feat_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, linear_out).
        """
        # x: (B, C, T, F)
        B, C, T, F = x.shape

        # 1) Batch normalization over (B, C, T, F)
        x = self.bn(x)

        # 2) Reorder to (B, T, C, F) and merge batch/time for pooling
        x = x.permute(0, 2, 1, 3)            # (B, T, C, F)
        x = x.reshape(B * T, C, F)          # (B*T, C, F)

        # 3) LPPool1d over the feature dimension -> (B*T, C, L)
        x = self.pool(x)

        # 4) Flatten channel and pooled-length into a single feature vector per time-step
        x = x.flatten(start_dim=1)          # (B*T, C * L)
        x = x.view(B, T, -1)                # (B, T, rnn_input_size)

        # 5) Process sequence with RNN (batch_first=True)
        rnn_out, h_n = self.rnn(x)          # rnn_out: (B, T, hidden); h_n: (num_layers, B, hidden)

        # 6) Aggregate: time-wise mean and last layer hidden state
        time_mean = rnn_out.mean(dim=1)     # (B, hidden)
        last_hidden = h_n[-1]               # (B, hidden) - take last layer's hidden state

        agg = torch.cat([time_mean, last_hidden], dim=1)  # (B, 2*hidden)

        # 7) Final non-linearity + linear projection
        out = self.relu(agg)
        out = self.fc(out)                  # (B, linear_out)

        return out

# Configuration / default parameters
batch_size = 8
channels = 3
seq_len = 16
feat_dim = 64
pool_kernel = 4
p = 2
rnn_hidden = 128
rnn_layers = 2
linear_out = 10

def get_inputs():
    """
    Returns a list containing the input tensor for the model:
      - x: (batch_size, channels, seq_len, feat_dim)
    """
    x = torch.randn(batch_size, channels, seq_len, feat_dim)
    return [x]

def get_init_inputs():
    """
    Returns the list of inputs required to initialize the Model in the same order as __init__:
      [channels, feat_dim, pool_kernel, p, rnn_hidden, rnn_layers, linear_out]
    """
    return [channels, feat_dim, pool_kernel, p, rnn_hidden, rnn_layers, linear_out]