import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex 3D-to-sequence model that:
    - Applies synchronized batch normalization over channels (SyncBatchNorm)
    - Performs adaptive 3D max-pooling to collapse spatial height/width and
      produce a fixed temporal length (AdaptiveMaxPool3d)
    - Interprets the pooled depth axis as a time dimension and processes it
      step-by-step with an RNNCell
    - Projects the final hidden state to an output feature vector

    This combines spatial reduction, normalization, and recurrent processing
    to produce a compact representation for the entire input volume.
    """
    def __init__(
        self,
        in_channels: int,
        pooled_depth: int = 8,
        hidden_size: int = 256,
        out_features: int = 128,
    ):
        """
        Args:
            in_channels (int): Number of input channels for the 3D volume.
            pooled_depth (int): Target temporal length after AdaptiveMaxPool3d.
            hidden_size (int): Hidden size of the RNNCell.
            out_features (int): Size of the final projected output vector.
        """
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.pooled_depth = pooled_depth
        self.hidden_size = hidden_size
        self.out_features = out_features

        # Normalize across channels for 5D input (N, C, D, H, W)
        self.sync_bn = nn.SyncBatchNorm(num_features=in_channels)

        # Reduce H and W to 1, set D to pooled_depth -> treat D as time dimension
        self.adaptive_pool = nn.AdaptiveMaxPool3d(output_size=(pooled_depth, 1, 1))

        # RNNCell to process each pooled depth slice; use ReLU non-linearity
        self.rnn_cell = nn.RNNCell(input_size=in_channels, hidden_size=hidden_size, nonlinearity='relu')

        # Final projection from hidden state to output features
        self.output_proj = nn.Linear(hidden_size, out_features)

        # Optional small layer norm on output for stability
        self.out_ln = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input 5D tensor of shape (batch, channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch, out_features).
        """
        # 1) Normalize channels
        x = self.sync_bn(x)

        # 2) Non-linear activation to increase expressivity
        x = torch.relu(x)

        # 3) Adaptive pooling -> (batch, channels, pooled_depth, 1, 1)
        x = self.adaptive_pool(x)

        # 4) Squeeze spatial dims and interpret depth as time:
        #    -> (batch, channels, pooled_depth)
        x = x.squeeze(-1).squeeze(-1)  # remove last two dims (1,1)

        # 5) Rearrange to (time, batch, channels) to iterate in temporal order
        x = x.permute(2, 0, 1)  # (pooled_depth, batch, in_channels)

        seq_len, batch_size, _ = x.shape

        # 6) Initialize hidden state (batch, hidden_size)
        hx = x.new_zeros(batch_size, self.hidden_size)

        # 7) Step through time with RNNCell
        for t in range(seq_len):
            # RNNCell expects (batch, input_size)
            hx = self.rnn_cell(x[t], hx)

        # 8) Project final hidden state to output features and normalize
        out = self.output_proj(hx)
        out = self.out_ln(out)
        return out

# Module-level configuration variables for test harness / input creation
batch_size = 8
in_channels = 32
depth = 32
height = 16
width = 16

pooled_depth = 8
hidden_size = 256
out_features = 128

def get_inputs():
    """
    Returns:
        list: [x] where x is a random tensor shaped (batch_size, in_channels, depth, height, width)
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Return initialization parameters for Model constructor in the same order:
        [in_channels, pooled_depth, hidden_size, out_features]
    """
    return [in_channels, pooled_depth, hidden_size, out_features]