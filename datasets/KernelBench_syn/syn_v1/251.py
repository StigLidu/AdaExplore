import torch
import torch.nn as nn

# Configuration / default parameters for the example
batch_size = 4
in_channels = 3
height = 8
width = 8

# Deconvolution 2D (lazy in_channels) will expand spatial dims from (H, W) -> (H2, W2)
deconv2d_out_channels = 16
deconv2d_kernel_size = 4
deconv2d_stride = 2
deconv2d_padding = 1  # chosen to produce clean doubling for these kernel/stride choices

# Deconvolution 1D (lazy in_channels) will operate on flattened spatial sequence
deconv1d_out_channels = 32
deconv1d_kernel_size = 3
deconv1d_stride = 1
deconv1d_padding = 1  # chosen to preserve sequence length

# RMSNorm epsilon
rms_eps = 1e-5

class Model(nn.Module):
    """
    Composite model that:
    - Uses a LazyConvTranspose2d to upsample spatial dimensions and change channel depth.
    - Flattens spatial dims to a sequence.
    - Applies RMSNorm over the feature dimension.
    - Uses a LazyConvTranspose1d to transform the sequence features.
    - Produces a pooled feature vector per batch item.

    This demonstrates combining lazy transposed convolutions with RMSNorm and
    reshaping/permutation patterns to bridge 2D <-> 1D computations.
    """
    def __init__(
        self,
        deconv2d_out_channels: int,
        deconv2d_kernel_size: int,
        deconv2d_stride: int,
        deconv1d_out_channels: int,
        deconv1d_kernel_size: int,
        deconv1d_stride: int,
        rms_eps: float = 1e-5
    ):
        super(Model, self).__init__()
        # LazyConvTranspose2d will infer in_channels on first forward pass.
        # We choose padding/output_padding to produce predictable spatial sizes for typical kernel/stride.
        self.deconv2d = nn.LazyConvTranspose2d(
            out_channels=deconv2d_out_channels,
            kernel_size=deconv2d_kernel_size,
            stride=deconv2d_stride,
            padding=deconv2d_padding,
            output_padding=0,
            bias=True
        )

        # RMSNorm normalizes over the feature dimension (the channel dimension after deconv2d).
        # normalized_shape is the number of channels output by deconv2d (known at init).
        self.rms = nn.RMSNorm(normalized_shape=deconv2d_out_channels, eps=rms_eps, elementwise_affine=True)

        # LazyConvTranspose1d will infer in_channels on first forward pass.
        # It operates on sequences where sequence length corresponds to flattened spatial dims.
        self.deconv1d = nn.LazyConvTranspose1d(
            out_channels=deconv1d_out_channels,
            kernel_size=deconv1d_kernel_size,
            stride=deconv1d_stride,
            padding=deconv1d_padding,
            output_padding=0,
            bias=True
        )

        # Small non-linearity before final pooling
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        1. Upsample spatially with LazyConvTranspose2d -> y: (B, V, H2, W2)
        2. Permute to (B, H2, W2, V) and flatten spatial dims -> seq: (B, T, V) where T=H2*W2
        3. Apply RMSNorm over the feature dimension V -> normed: (B, T, V)
        4. Transpose to (B, V, T) to feed LazyConvTranspose1d -> z: (B, O, T)
        5. Apply non-linearity and pool over time to get (B, O)
        """
        # Step 1: 2D transposed convolution (upsampling)
        y = self.deconv2d(x)  # (B, V, H2, W2)

        # Step 2: move channels to last dim and flatten spatial dims into sequence
        y_perm = y.permute(0, 2, 3, 1)  # (B, H2, W2, V)
        B, H2, W2, V = y_perm.shape
        seq = y_perm.contiguous().view(B, H2 * W2, V)  # (B, T, V) where T = H2 * W2

        # Step 3: RMS normalization over the feature (V) dimension
        normed = self.rms(seq)  # (B, T, V)

        # Step 4: prepare for 1D transposed conv: (B, V, T)
        conv1d_in = normed.transpose(1, 2).contiguous()  # (B, V, T)
        z = self.deconv1d(conv1d_in)  # (B, O, T_out) where O = deconv1d_out_channels

        # Step 5: activation and temporal pooling -> final per-batch feature vector
        z = self.act(z)  # (B, O, T_out)
        out = z.mean(dim=-1)  # global average over time -> (B, O)
        return out

def get_inputs():
    """
    Returns a list of input tensors for the model's forward().

    The input shape matches (batch_size, in_channels, height, width).
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization arguments for Model() in the same order as its __init__ signature.
    """
    return [
        deconv2d_out_channels,
        deconv2d_kernel_size,
        deconv2d_stride,
        deconv1d_out_channels,
        deconv1d_kernel_size,
        deconv1d_stride,
        rms_eps,
    ]