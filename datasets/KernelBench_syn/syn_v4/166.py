import torch
import torch.nn as nn

# Module-level configuration
batch_size = 8
in_channels = 3
out_channels = 6
depth = 4
height = 4
width = 4
target_seq_len = 32  # length after AdaptiveAvgPool1d

class Model(nn.Module):
    """
    Complex module that:
    - Upsamples a 3D feature volume using ConvTranspose3d
    - Applies a LogSigmoid non-linearity
    - Collapses spatial 3D dims into a 1D sequence and applies AdaptiveAvgPool1d
    - Scales channels by a learned per-channel parameter and produces a final 2D (N, seq_len) output

    Input shape: (N, in_channels, D, H, W)
    Output shape: (N, target_seq_len)
    """
    def __init__(
        self,
        in_ch: int = in_channels,
        out_ch: int = out_channels,
        seq_len: int = target_seq_len,
        kernel_size: int = 3,
        stride: int = 2,
    ):
        super(Model, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.seq_len = seq_len

        # Transposed convolution to increase spatial resolution (D,H,W)
        # Using padding=1 and output_padding=1 with stride=2 and kernel=3 typically doubles each spatial dim when input dims are small integers.
        self.deconv = nn.ConvTranspose3d(
            in_channels=self.in_ch,
            out_channels=self.out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=1,
            output_padding=1,
            bias=True
        )

        # Non-linearity applied elementwise after deconv
        self.logsig = nn.LogSigmoid()

        # Adaptive 1D average pooling applied after flattening spatial dims into a sequence
        self.pool = nn.AdaptiveAvgPool1d(self.seq_len)

        # Learnable per-channel scale (will broadcast over batch and sequence length)
        # Shape: (out_ch, 1) so broadcasting yields (1, out_ch, seq_len)
        self.channel_scale = nn.Parameter(torch.ones(self.out_ch, 1))

        # Small epsilon for numerical stability in final normalization
        self.eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Steps:
        1. Apply ConvTranspose3d to upsample the input volume.
        2. Apply LogSigmoid activation element-wise.
        3. Reshape spatial dims (D,H,W) into a single length L and treat as (N, C, L).
        4. Apply AdaptiveAvgPool1d to reduce L -> seq_len.
        5. Scale per-channel activations by a learned parameter.
        6. Compute channel-wise normalized output by dividing by channel-wise L2 norm,
           then aggregate channels by mean to produce the final (N, seq_len) tensor.

        Args:
            x (torch.Tensor): Input tensor with shape (N, in_ch, D, H, W)

        Returns:
            torch.Tensor: Output tensor with shape (N, seq_len)
        """
        # 1) Upsample spatial volume
        y = self.deconv(x)  # (N, out_ch, D2, H2, W2)

        # 2) Non-linearity
        y = self.logsig(y)  # (N, out_ch, D2, H2, W2)

        # 3) Collapse spatial dims into a single sequence dimension
        N, C, D2, H2, W2 = y.shape
        seq_len_in = D2 * H2 * W2
        y = y.view(N, C, seq_len_in)  # (N, out_ch, L)

        # 4) Adaptive average pooling to fixed length
        y = self.pool(y)  # (N, out_ch, seq_len)

        # 5) Channel scaling
        # channel_scale: (out_ch, 1) -> (1, out_ch, 1) via unsqueeze
        scale = self.channel_scale.unsqueeze(0)  # (1, out_ch, 1)
        y = y * scale  # broadcast multiply -> (N, out_ch, seq_len)

        # 6) Channel-wise normalization then aggregate channels
        # Compute L2 norm across sequence for each channel: (N, out_ch, 1)
        channel_norm = torch.sqrt(torch.sum(y * y, dim=2, keepdim=True) + self.eps)
        y_normalized = y / channel_norm  # (N, out_ch, seq_len)

        # Aggregate across channels to produce final output
        out = torch.mean(y_normalized, dim=1)  # (N, seq_len)

        return out

def get_inputs():
    """
    Returns a list containing the input tensor for the model.

    Input shape follows the module-level config: (batch_size, in_channels, depth, height, width)
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for the Model constructor.

    This allows external code to construct the Model with the same settings used here.
    """
    return [in_channels, out_channels, target_seq_len]