import torch
import torch.nn as nn
import torch.nn.functional as F

# Configuration variables
BATCH = 8
BIL_IN1 = 128
BIL_IN2 = 256
BIL_OUT = 64

CONV_OUT_CHANNELS = 32
CONV_KERNEL = 4  # kernel size for ConvTranspose1d
CONV_STRIDE = 2

VOL_CHANNELS = 16
VOL_D = 8
VOL_H = 16
VOL_W = 16
POOL_KERNEL = (2, 2, 2)

class Model(nn.Module):
    """
    Complex composite module that:
    - Applies a Bilinear transform to two vector inputs.
    - Uses the bilinear output as a latent "channel" input to a LazyConvTranspose1d (1D deconvolution),
      producing a temporal feature map.
    - Applies AvgPool3d to a separate volumetric input and projects its pooled channel summaries
      to gate the deconvolution features.
    The final output is a gated fusion of the deconvolution features.
    """
    def __init__(
        self,
        bil_in1: int = BIL_IN1,
        bil_in2: int = BIL_IN2,
        bil_out: int = BIL_OUT,
        conv_out_channels: int = CONV_OUT_CHANNELS,
        conv_kernel: int = CONV_KERNEL,
        conv_stride: int = CONV_STRIDE,
        pool_kernel = POOL_KERNEL,
        vol_channels: int = VOL_CHANNELS,
    ):
        """
        Initializes submodules and learnable parameters.

        Args:
            bil_in1: size of first bilinear input features
            bil_in2: size of second bilinear input features
            bil_out: bilinear output features (will be used as conv-in channels lazily)
            conv_out_channels: number of output channels for ConvTranspose1d
            conv_kernel: kernel size for ConvTranspose1d
            conv_stride: stride for ConvTranspose1d
            pool_kernel: kernel size for AvgPool3d
            vol_channels: number of channels in volumetric input
        """
        super(Model, self).__init__()
        # Bilinear transforms two vectors into a joint representation
        self.bilinear = nn.Bilinear(bil_in1, bil_in2, bil_out)

        # Lazy deconvolution: in_channels will be inferred at first forward from bilinear output channels
        # We provide out_channels and kernel_size explicitly.
        self.deconv = nn.LazyConvTranspose1d(
            out_channels=conv_out_channels,
            kernel_size=conv_kernel,
            stride=conv_stride,
            padding=0,
            bias=True
        )

        # 3D average pooling over volumetric input
        self.avgpool3d = nn.AvgPool3d(kernel_size=pool_kernel, stride=pool_kernel)

        # Learnable projection from pooled volumetric channels to conv_out_channels,
        # used to gate the deconvolution features. Shape: (vol_channels) x (conv_out_channels)
        self.gate_proj = nn.Parameter(torch.randn(vol_channels, conv_out_channels))

        # Small epsilon for numerical stability in gating
        self.eps = 1e-6

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, vol: torch.Tensor) -> torch.Tensor:
        """
        Forward computation combining bilinear, deconvolution, and pooled gating.

        Args:
            x1: Tensor of shape (N, bil_in1)
            x2: Tensor of shape (N, bil_in2)
            vol: Volumetric tensor of shape (N, vol_channels, D, H, W)

        Returns:
            Tensor of shape (N, conv_out_channels) representing fused features.
        """
        # 1) Bilinear interaction and non-linearity -> (N, bil_out)
        bil = self.bilinear(x1, x2)
        bil = F.relu(bil)

        # 2) Prepare bilinear output as (N, C_in, L=1) for ConvTranspose1d
        #    This will initialize LazyConvTranspose1d's in_channels on first call.
        deconv_in = bil.unsqueeze(-1)  # shape: (N, bil_out, 1)

        # 3) 1D transposed convolution expands temporal dimension: (N, conv_out_channels, L_out)
        deconv_out = self.deconv(deconv_in)

        # 4) Reduce temporal dimension by mean to get a per-channel descriptor: (N, conv_out_channels)
        deconv_feat = deconv_out.mean(dim=2)

        # 5) Pool the volumetric input and compute channel-wise summaries: (N, vol_channels)
        pooled = self.avgpool3d(vol)  # shape: (N, vol_channels, D', H', W')
        pooled_mean = pooled.mean(dim=[2, 3, 4])  # (N, vol_channels)

        # 6) Project pooled summaries into conv_out_channels via learnable linear map (matrix multiply)
        #    pooled_mean @ gate_proj -> (N, conv_out_channels)
        gate_logits = pooled_mean @ self.gate_proj  # (N, conv_out_channels)

        # 7) Sigmoid gating and fusion with deconv features (elementwise): gated * deconv + residual
        gate = torch.sigmoid(gate_logits)
        fused = deconv_feat * gate + deconv_feat  # residual-style fusion

        return fused

# Functions to provide inputs to the module (and initialization parameters)

def get_inputs():
    """
    Returns a list of input tensors matching the forward signature of Model:
    [x1, x2, vol]
    """
    x1 = torch.randn(BATCH, BIL_IN1)
    x2 = torch.randn(BATCH, BIL_IN2)
    vol = torch.randn(BATCH, VOL_CHANNELS, VOL_D, VOL_H, VOL_W)
    return [x1, x2, vol]

def get_init_inputs():
    """
    Returns initialization parameters for constructing the Model with defaults matching
    the top-level configuration variables. This can be used to instantiate the Model
    programmatically if needed.
    """
    return [BIL_IN1, BIL_IN2, BIL_OUT, CONV_OUT_CHANNELS, CONV_KERNEL, CONV_STRIDE, POOL_KERNEL, VOL_CHANNELS]