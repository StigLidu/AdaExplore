import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex model that:
      - Processes 1D signals with a lazy-initialized Conv1d
      - Applies LPPool1d to reduce temporal resolution
      - Treats the pooled 1D signal as a single-row 2D feature map and applies a lazy ConvTranspose2d
        to perform learned upsampling along the temporal axis
      - Uses a final lazy pointwise Conv1d to mix channels and produce the desired output channels
      - Combines activations and a residual-like skip through a channel-wise concatenation before final projection

    This demonstrates mixing nn.LazyConv1d, nn.LPPool1d and nn.LazyConvTranspose2d in a single module.
    """
    def __init__(
        self,
        conv1_out_channels: int = 32,
        conv1_kernel: int = 7,
        pool_p: float = 2.0,
        pool_kernel: int = 4,
        convtr_out_channels: int = 16,
        convtr_kernel: int = 5,
        convtr_stride: int = 2,
        final_out_channels: int = 1,
    ):
        """
        Initializes the composite module.

        Args:
            conv1_out_channels: number of output channels for the initial Conv1d.
            conv1_kernel: kernel size for the initial Conv1d (odd recommended for 'same' padding).
            pool_p: the p-norm for LPPool1d.
            pool_kernel: kernel size / stride for LPPool1d.
            convtr_out_channels: number of output channels for ConvTranspose2d.
            convtr_kernel: kernel size along temporal axis for ConvTranspose2d (spatial dim is 1).
            convtr_stride: stride along temporal axis for ConvTranspose2d (controls upsampling).
            final_out_channels: channels produced by final pointwise Conv1d.
        """
        super(Model, self).__init__()

        # Lazy Conv1d: in_channels will be inferred at first forward call
        self.conv1 = nn.LazyConv1d(out_channels=conv1_out_channels, kernel_size=conv1_kernel, padding=conv1_kernel // 2)

        # LPPool1d pooling reduces temporal resolution with a power-average
        self.pool = nn.LPPool1d(norm_type=pool_p, kernel_size=pool_kernel, stride=pool_kernel, ceil_mode=False)

        # ConvTranspose2d performs learned upsampling along the temporal axis.
        # We use kernel (1, convtr_kernel) and stride (1, convtr_stride) so only width dimension is affected.
        # in_channels is lazy; we set out_channels explicitly.
        padding_w = convtr_kernel // 2
        output_padding_w = convtr_stride - 1 if convtr_stride > 1 else 0
        self.convtr = nn.LazyConvTranspose2d(
            out_channels=convtr_out_channels,
            kernel_size=(1, convtr_kernel),
            stride=(1, convtr_stride),
            padding=(0, padding_w),
            output_padding=(0, output_padding_w)
        )

        # Final pointwise Conv1d to mix channels into desired output channels
        self.final_proj = nn.LazyConv1d(out_channels=final_out_channels, kernel_size=1)

        # Nonlinearities and normalization
        self.act = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(conv1_out_channels)  # will error if conv1_out_channels != actual set later; kept for clarity
        # Note: BatchNorm1d expects a fixed number of channels. We will not call it until after conv1 instantiated,
        # and PyTorch will accept the channel number as provided above.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (N, C_in, L) where C_in is inferred lazily (commonly 1).

        Returns:
            Tensor of shape (N, final_out_channels, L_out) after the pipeline.
        """
        # Initial convolution -> (N, C1, L)
        conv1_out = self.conv1(x)
        # Apply activation
        conv1_out = self.act(conv1_out)

        # BatchNorm over channels (safe because conv1_out_channels is known)
        # If the declared conv1_out_channels differs from actual weight shape, BatchNorm will still work if shapes match.
        try:
            conv1_out = self.bn1(conv1_out)
        except Exception:
            # In case the provided bn1 channel count doesn't match (defensive), skip BN.
            pass

        # Preserve a skip / high-resolution feature by average pooling along time to match later shapes if needed.
        # Perform LPPool1d to downsample temporal axis -> (N, C1, Lp)
        pooled = self.pool(conv1_out)

        # Interpret the pooled 1D as a single-row 2D feature map: (N, C1, 1, Lp)
        pooled_2d = pooled.unsqueeze(2)

        # ConvTranspose2d upsamples along width -> (N, C2, 1, Lout)
        up_2d = self.convtr(pooled_2d)
        up_2d = self.act(up_2d)

        # Squeeze the height dimension -> (N, C2, Lout)
        up_1d = up_2d.squeeze(2)

        # To create a richer representation, concatenate a pooled-and-broadcasted version of the early feature maps.
        # Project conv1_out to same time resolution as up_1d by average pooling along temporal dim.
        # We use adaptive average pooling to get the exact length.
        target_len = up_1d.size(-1)
        if target_len <= 0:
            raise ValueError("Upsampled length must be positive")
        conv1_resampled = F.adaptive_avg_pool1d(conv1_out, output_size=target_len)

        # Concatenate along the channel axis: (N, C2 + C1, Lout)
        enriched = torch.cat([up_1d, conv1_resampled], dim=1)

        # Final projection to desired output channels via pointwise Conv1d
        out = self.final_proj(enriched)
        # Final activation (tanh to keep outputs bounded)
        out = torch.tanh(out)
        return out

# Configuration / default parameters
batch_size = 8
input_channels = 1
input_length = 16384  # 1D signal length
conv1_out_channels = 32
conv1_kernel = 7
pool_p = 2.0
pool_kernel = 4
convtr_out_channels = 16
convtr_kernel = 5
convtr_stride = 2
final_out_channels = 1

def get_inputs():
    """
    Returns example input tensors for the model.
    Input shape: (batch_size, input_channels, input_length)
    """
    x = torch.randn(batch_size, input_channels, input_length)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters for the Model constructor in the same order.
    """
    return [
        conv1_out_channels,
        conv1_kernel,
        pool_p,
        pool_kernel,
        convtr_out_channels,
        convtr_kernel,
        convtr_stride,
        final_out_channels,
    ]