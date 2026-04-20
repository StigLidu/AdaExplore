import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex sequence-to-vector module that combines 1D padding, adaptive max pooling,
    channel-wise gating, and two lazy-initialized 2D convolutions.

    Overall computation:
      1. ConstantPad1d on the time dimension.
      2. AdaptiveMaxPool1d to reduce the time dimension to a fixed length.
      3. Compute a channel-wise gate from the pooled features (global average + sigmoid).
      4. Apply the gate to modulate the pooled sequence.
      5. Insert a singleton spatial dimension and process with two LazyConv2d layers:
         - A wider (1 x kernel_size) conv to capture temporal patterns across channels.
         - A 1x1 conv to project input channels to the same out_channels (used for a learned residual).
      6. Fuse the conv outputs, apply ReLU, and finally adaptively pool to a vector per batch.
    The lazy convolutions allow the model to accept inputs with unknown in_channels at construction time.
    """
    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        pad: int,
        pad_value: float,
        pool_output_size: int,
    ):
        """
        Args:
            out_channels (int): Number of output channels for the conv layers.
            kernel_size (int): Temporal kernel size for the 1xK conv.
            pad (int): Number of values to pad on both sides of the time dimension.
            pad_value (float): Value used by ConstantPad1d for padding.
            pool_output_size (int): Sequence length after the first adaptive max pooling.
        """
        super(Model, self).__init__()

        # Pad the 1D temporal input on both sides with a constant value
        self.pad = nn.ConstantPad1d(pad, pad_value)

        # Reduce/normalize the temporal length to a fixed size
        self.pool = nn.AdaptiveMaxPool1d(pool_output_size)

        # Lazy 2D conv to capture cross-channel x temporal patterns.
        # Input will be shaped (N, C_in, 1, pool_output_size); kernel (1, kernel_size).
        self.conv_temporal = nn.LazyConv2d(
            out_channels=out_channels,
            kernel_size=(1, kernel_size),
            stride=(1, 1),
            padding=(0, kernel_size // 2),
            bias=True,
        )

        # Lazy 1x1 conv to project the original channels to out_channels for a learned residual connection.
        self.project_1x1 = nn.LazyConv2d(out_channels=out_channels, kernel_size=1, bias=True)

        # Final pooling to collapse the temporal dimension to a vector per channel
        self.final_pool = nn.AdaptiveMaxPool1d(1)

        # Small epsilon for numerical stability if needed in gating
        self.eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, channels, seq_len).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, out_channels).
        """
        # Step 1: Pad the time dimension
        # x: (N, C, L)
        x_padded = self.pad(x)

        # Step 2: Adaptive max pool to fixed temporal length
        x_pooled = self.pool(x_padded)  # (N, C, Lp)

        # Step 3: Channel-wise gating: global average over time -> sigmoid gate
        channel_avg = x_pooled.mean(dim=2, keepdim=True)  # (N, C, 1)
        gate = torch.sigmoid(channel_avg)  # (N, C, 1)

        # Step 4: Apply gate to modulate the pooled sequence (broadcast over temporal dim)
        x_modulated = x_pooled * gate  # (N, C, Lp)

        # Step 5: Insert spatial singleton dimension to use 2D convs
        x_spatial = x_modulated.unsqueeze(2)  # (N, C, 1, Lp)

        # Apply temporal conv (1 x kernel_size) to capture patterns across time for each channel mix
        conv_out = self.conv_temporal(x_spatial)  # (N, out_channels, 1, Lp)

        # Project original inputs to out_channels to build a learned residual before activation
        proj_out = self.project_1x1(x_spatial)  # (N, out_channels, 1, Lp)

        # Fuse, non-linearity
        fused = F.relu(conv_out + proj_out)  # (N, out_channels, 1, Lp)

        # Remove the singleton spatial dim -> (N, out_channels, Lp)
        fused = fused.squeeze(2)

        # Step 6: Final temporal pooling to produce a vector per batch
        out_vec = self.final_pool(fused).squeeze(2)  # (N, out_channels)

        return out_vec

# Configuration / default sizes
batch_size = 8
in_channels = 13
seq_len = 512

out_channels = 32
kernel_size = 9
pad = 4  # pad both sides equally
pad_value = 0.0
pool_output_size = 64

def get_inputs():
    """
    Returns a list with one input tensor of shape (batch_size, in_channels, seq_len).
    """
    x = torch.randn(batch_size, in_channels, seq_len)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters matching Model.__init__:
      [out_channels, kernel_size, pad, pad_value, pool_output_size]
    """
    return [out_channels, kernel_size, pad, pad_value, pool_output_size]