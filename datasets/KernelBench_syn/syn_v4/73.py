import torch
import torch.nn as nn

# Configuration (module-level)
batch_size = 8
in_channels = 16
mid_channels = 32
out_channels = 24  # final channel dimension after linear mixing
seq_len = 64

kernel_size = 5
stride = 2
padding = 1
output_padding = 1
dilation = 1
use_bias = True
leaky_slope = 0.05

class Model(nn.Module):
    """
    A composite 1D upsampling + channel-mixing module that demonstrates:
      - ConvTranspose1d for learned upsampling
      - Non-linearities ReLU6 and LeakyReLU applied in sequence
      - A learnable channel-mixing matrix applied via batched matmul
      - A residual-style skip connection after reshaping and mixing

    Forward pattern:
      x (B, C_in, L) ->
      convt -> (B, C_mid, L_up) ->
      ReLU6 ->
      permute to (B, L_up, C_mid) ->
      batch matmul with W (C_mid, C_out) -> (B, L_up, C_out) ->
      LeakyReLU ->
      permute back to (B, C_out, L_up) and add spatially-broadcasted skip connection
    """
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
        leaky_slope: float = 0.01
    ):
        super(Model, self).__init__()
        # Transposed convolution upsamples the temporal dimension and increases channels
        self.convt = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            bias=bias
        )

        # Non-linearities
        self.relu6 = nn.ReLU6(inplace=False)
        self.leaky = nn.LeakyReLU(negative_slope=leaky_slope, inplace=False)

        # Learnable channel mixing matrix W: maps mid_channels -> out_channels
        # Registered as a Parameter so it will be updated by optimizers
        self.W = nn.Parameter(torch.empty(mid_channels, out_channels))
        nn.init.kaiming_uniform_(self.W, a=leaky_slope)

        # If channel dimensions match, allow an identity skip; otherwise provide a learned projection
        if in_channels != out_channels:
            # small 1x1 conv implemented as ConvTranspose1d with kernel_size=1,stride=1 to project skip to out_channels
            self.skip_proj = nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            )
        else:
            self.skip_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation.

        Args:
            x: Tensor of shape (B, C_in, L)

        Returns:
            Tensor of shape (B, C_out, L_up)
        """
        # 1) Learned upsampling and channel expansion
        y = self.convt(x)  # (B, C_mid, L_up)

        # 2) Bounded ReLU to clip activations (helps numerical stability)
        y = self.relu6(y)

        # 3) Prepare for channel mixing: (B, C_mid, L_up) -> (B, L_up, C_mid)
        y_perm = y.permute(0, 2, 1)

        # 4) Channel mixing with a learned matrix W: (B, L_up, C_mid) @ (C_mid, C_out) -> (B, L_up, C_out)
        mixed = torch.matmul(y_perm, self.W)

        # 5) Non-linearity after mixing
        mixed = self.leaky(mixed)

        # 6) Permute back to (B, C_out, L_up)
        out = mixed.permute(0, 2, 1)

        # 7) Create skip connection: project original input to out_channels and upsample spatially if needed
        #    To match the temporal dimension (L_up), we upsample x using nearest interpolation before projection if necessary.
        L_up = out.shape[-1]
        if x.shape[-1] != L_up:
            # nearest upsample along temporal dimension
            x_upsampled = torch.nn.functional.interpolate(x, size=L_up, mode='nearest')
        else:
            x_upsampled = x

        if self.skip_proj is not None:
            skip = self.skip_proj(x_upsampled)
        else:
            skip = x_upsampled  # already same channels

        # 8) Residual addition and final clamp to keep values bounded
        res = out + skip
        # Use ReLU6 again to bound final outputs
        final = self.relu6(res)
        return final

# Initialization configuration values returned by get_init_inputs()
init_in_channels = in_channels
init_mid_channels = mid_channels
init_out_channels = out_channels
init_kernel_size = kernel_size
init_stride = stride
init_padding = padding
init_output_padding = output_padding
init_dilation = dilation
init_bias = use_bias
init_leaky_slope = leaky_slope

def get_inputs():
    """
    Returns a list with a single input tensor shaped (batch_size, in_channels, seq_len).
    """
    x = torch.randn(batch_size, in_channels, seq_len)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters for Model.__init__ in the same order.
    """
    return [
        init_in_channels,
        init_mid_channels,
        init_out_channels,
        init_kernel_size,
        init_stride,
        init_padding,
        init_output_padding,
        init_dilation,
        init_bias,
        init_leaky_slope
    ]