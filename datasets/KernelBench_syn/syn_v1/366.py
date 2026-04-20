import torch
import torch.nn as nn
import torch.nn.functional as F

# Module-level configuration
batch_size = 8
in_channels = 4
seq_len = 64
default_out_channels = 16
default_kernel_size = 4
default_stride = 2
default_padding = 1
default_scale = 0.5

class Model(nn.Module):
    """
    Complex 1D upsampling block that demonstrates a combined use of:
      - LazyConvTranspose1d (lazy initialization of in_channels),
      - SyncBatchNorm for synchronized normalization,
      - GLU gating non-linearity,
      - a small non-linear post-scaling and a second BatchNorm.

    The forward pass performs:
      1) A transposed convolution to increase temporal resolution and produce 2*C channels
      2) A SyncBatchNorm over the produced channels
      3) A GLU gate that splits channels in half and gates them (reducing to C channels)
      4) A tanh non-linearity followed by a learned scalar scaling
      5) A final SyncBatchNorm for stabilized outputs

    This produces an upsampled and gated feature map suitable as a decoder/residual building block.
    """
    def __init__(
        self,
        out_channels: int = default_out_channels,
        kernel_size: int = default_kernel_size,
        stride: int = default_stride,
        padding: int = default_padding,
        scale: float = default_scale,
    ):
        """
        Args:
            out_channels (int): Number of output channels after GLU (i.e., effective channels).
            kernel_size (int): Kernel size for the transposed convolution.
            stride (int): Stride for the transposed convolution (controls upsampling).
            padding (int): Padding for the transposed convolution.
            scale (float): Initial scalar multiplier applied after non-linearity.
        """
        super(Model, self).__init__()

        # We request 2 * out_channels from the transposed conv so that GLU
        # (which halves the channel dim) yields exactly out_channels.
        self.out_channels = out_channels
        self.convt = nn.LazyConvTranspose1d(
            out_channels=out_channels * 2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        # BatchNorm layers for stabilization; first operates on 2*C channels produced by convt,
        # second operates on C channels after GLU.
        self.bn1 = nn.SyncBatchNorm(num_features=out_channels * 2)
        self.bn2 = nn.SyncBatchNorm(num_features=out_channels)

        # GLU gate along channel dimension (1 for (N, C, L) layout)
        self.glu = nn.GLU(dim=1)

        # Learnable scalar parameter to scale the non-linearity output
        self.scale = nn.Parameter(torch.tensor(float(scale), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, seq_len)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, new_seq_len),
                          where new_seq_len is determined by the transposed convolution params.
        """
        # 1) Transposed convolution upsamples temporal dimension and emits 2*C channels
        y = self.convt(x)  # shape: (N, 2*C, L_out)

        # 2) First BatchNorm across the produced channels
        y = self.bn1(y)

        # 3) GLU halves channels (gated linear unit), producing (N, C, L_out)
        y = self.glu(y)

        # 4) Non-linearity and learned scaling. Using tanh to bound activations then scale.
        y = torch.tanh(y) * self.scale

        # 5) Final BatchNorm for stabilized outputs
        y = self.bn2(y)

        return y

def get_inputs():
    """
    Returns example input tensors matching the expected input signature of Model.forward.
    We provide a 1D sequence input: (batch_size, in_channels, seq_len)
    """
    x = torch.randn(batch_size, in_channels, seq_len)
    return [x]

def get_init_inputs():
    """
    Returns the initialization arguments for Model in the same order as __init__:
      [out_channels, kernel_size, stride, padding, scale]
    Matching the module-level defaults above.
    """
    return [default_out_channels, default_kernel_size, default_stride, default_padding, default_scale]