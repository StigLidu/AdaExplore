import torch
import torch.nn as nn

"""
Complex module that mixes 3D circular padding, randomized leaky ReLU activation,
and a transformation that leverages 1D adaptive average pooling across the
width dimension after folding depth and height into the channel axis.

The forward pass produces a per-channel log-probability vector (log-softmax)
for each batch element by:
  1. Circularly padding the 3D spatial input.
  2. Applying a randomized leaky ReLU (RReLU).
  3. Folding depth and height into an expanded channel axis and pooling the width
     dimension adaptively (AdaptiveAvgPool1d).
  4. Aggregating pooled statistics to produce per-channel gating values.
  5. Re-weighting global per-channel means by the gates and returning
     a log-softmax over channels.
"""

class Model(nn.Module):
    def __init__(self, padding: tuple, pool_output_size: int):
        """
        Args:
            padding (tuple): 6-int tuple for CircularPad3d (left, right, top, bottom, front, back).
            pool_output_size (int): output length for AdaptiveAvgPool1d applied on the width axis.
        """
        super(Model, self).__init__()
        # Circular padding for 3D inputs (N, C, D, H, W)
        self.circular_pad = nn.CircularPad3d(padding)
        # Randomized leaky ReLU activation with typical lower/upper bounds
        self.rrelu = nn.RReLU(lower=0.125, upper=0.333, inplace=False)
        # AdaptiveAvgPool1d that will be applied after reshaping to (N, C * (D*H), W)
        self.adaptive_pool1d = nn.AdaptiveAvgPool1d(pool_output_size)
        # store pool_output_size for potential debugging/inspection
        self.pool_output_size = pool_output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, D, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (N, C) representing per-channel
                          log-probabilities (log-softmax across channels).
        """
        # Step 1: Circular pad the 3D spatial tensor
        x_padded = self.circular_pad(x)  # (N, C, D_p, H_p, W_p)

        # Step 2: Apply randomized leaky ReLU activation
        x_activated = self.rrelu(x_padded)  # same shape as x_padded

        N, C, Dp, Hp, Wp = x_activated.shape

        # Step 3: Fold depth and height into an expanded channel axis to pool width
        # New shape for pooling: (N, C * (Dp * Hp), Wp)
        folded_channels = C * (Dp * Hp)
        x_for_pool = x_activated.reshape(N, folded_channels, Wp)

        # Adaptive average pool along the width dimension to a fixed length
        x_pooled_w = self.adaptive_pool1d(x_for_pool)  # (N, folded_channels, pool_output_size)

        # Step 4: Reshape pooled output back to (N, C, Dp*Hp, pool_output_size)
        x_pooled_reshaped = x_pooled_w.view(N, C, Dp * Hp, self.pool_output_size)

        # Aggregate pooled statistics:
        #   - mean across the pooled width dimension -> (N, C, Dp*Hp)
        pooled_mean_over_width = x_pooled_reshaped.mean(dim=-1)

        #   - then mean across the spatial folded dimension (Dp*Hp) -> (N, C)
        channel_statistic = pooled_mean_over_width.mean(dim=-1)

        # Step 5: Compute global per-channel means from the activated (unpadded spatial)
        # Use the activated padded tensor's spatial dims for a robust per-channel statistic
        global_channel_mean = x_activated.mean(dim=[2, 3, 4])  # (N, C)

        # Create gates from the pooled-derived statistics and apply to global means
        gates = torch.sigmoid(channel_statistic)  # (N, C)
        gated_channels = global_channel_mean * gates  # (N, C)

        # Final normalization: log-softmax across channels to produce log-probabilities
        output = torch.log_softmax(gated_channels, dim=1)  # (N, C)

        return output

# Configuration (module-level)
batch_size = 8
channels = 16
depth = 6
height = 6
width = 32
# Padding for CircularPad3d: (left, right, top, bottom, front, back)
padding = (1, 1, 2, 2, 0, 0)
pool_output_size = 10

def get_inputs():
    """
    Returns a list containing the primary input tensor(s) to the model.

    The input shape is (batch_size, channels, depth, height, width).
    """
    x = torch.randn(batch_size, channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters required to construct the Model instance:
    [padding, pool_output_size]
    """
    return [padding, pool_output_size]