import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex model that demonstrates a combination of ConvTranspose2d, LocalResponseNorm,
    and InstanceNorm1d used in a non-trivial dataflow. The pipeline is:

      1. Transposed convolution to upsample feature maps.
      2. Local response normalization across channels.
      3. Reshape spatial dims into a 1D sequence per channel and apply InstanceNorm1d.
      4. SELU nonlinearity.
      5. Spatial aggregation (mean) and final linear projection via a learnable matrix.

    This creates a multi-stage normalization + upsampling pattern useful for generative or
    decoder-like networks.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        output_padding: int,
        lrn_size: int,
        lrn_alpha: float,
        lrn_beta: float,
        lrn_k: float,
        instnorm_affine: bool,
        proj_out_dim: int
    ):
        """
        Initializes the composite module.

        Args:
            in_channels (int): Number of input channels to ConvTranspose2d.
            out_channels (int): Number of output channels from ConvTranspose2d and features for InstanceNorm1d.
            kernel_size (int): Kernel size for ConvTranspose2d.
            stride (int): Stride for ConvTranspose2d.
            padding (int): Padding for ConvTranspose2d.
            dilation (int): Dilation for ConvTranspose2d.
            output_padding (int): Output padding for ConvTranspose2d.
            lrn_size (int): LocalResponseNorm window size.
            lrn_alpha (float): LocalResponseNorm alpha.
            lrn_beta (float): LocalResponseNorm beta.
            lrn_k (float): LocalResponseNorm k (bias).
            instnorm_affine (bool): If True, InstanceNorm1d has learnable affine params.
            proj_out_dim (int): Output dimension for the final projection.
        """
        super(Model, self).__init__()
        # Transposed convolution to upsample spatial resolution while changing channels
        self.deconv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            bias=False
        )

        # Local response normalization across channels (keeps spatial dims)
        self.lrn = nn.LocalResponseNorm(size=lrn_size, alpha=lrn_alpha, beta=lrn_beta, k=lrn_k)

        # InstanceNorm1d works on (N, C, L) where L is sequence length;
        # we'll reshape spatial dims into L = H_out * W_out
        self.instnorm = nn.InstanceNorm1d(num_features=out_channels, affine=instnorm_affine)

        # Final projection matrix as a learnable parameter (C -> proj_out_dim)
        self.proj = nn.Parameter(torch.randn(out_channels, proj_out_dim))
        self.proj_bias = nn.Parameter(torch.randn(proj_out_dim))

        # Activation
        self.activation = torch.selu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor with shape (N, in_channels, H, W).

        Returns:
            torch.Tensor: Output tensor with shape (N, proj_out_dim).
        """
        # 1) Upsample + channel transform
        y = self.deconv(x)  # shape: (N, out_channels, H_out, W_out)

        # 2) Local response normalization (across channels)
        y = self.lrn(y)  # same shape as input

        # 3) Reshape spatial dims into a sequence for InstanceNorm1d: (N, C, L)
        N, C, H_out, W_out = y.shape
        y_seq = y.view(N, C, H_out * W_out)  # (N, C, L)

        # 4) Instance normalization per sample per channel across spatial positions
        y_norm = self.instnorm(y_seq)  # (N, C, L)

        # 5) Nonlinearity
        y_act = self.activation(y_norm)  # (N, C, L)

        # 6) Aggregate spatially (mean) -> (N, C)
        y_pooled = y_act.mean(dim=2)  # (N, C)

        # 7) Final linear projection using learned parameter matrix -> (N, proj_out_dim)
        out = torch.matmul(y_pooled, self.proj) + self.proj_bias  # (N, proj_out_dim)
        return out


# Configuration variables
batch_size = 8
in_channels = 3
out_channels = 16
height = 32
width = 32

kernel_size = 4
stride = 2
padding = 1
dilation = 1
output_padding = 0

lrn_size = 5
lrn_alpha = 1e-4
lrn_beta = 0.75
lrn_k = 1.0

instnorm_affine = True
proj_out_dim = 64


def get_inputs():
    """
    Returns a list with a single input tensor matching the configured shape:
    (batch_size, in_channels, height, width)
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    """
    Returns initialization parameters in the same order as Model.__init__ arguments.
    """
    return [
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        output_padding,
        lrn_size,
        lrn_alpha,
        lrn_beta,
        lrn_k,
        instnorm_affine,
        proj_out_dim
    ]