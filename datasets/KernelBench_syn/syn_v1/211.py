import torch
import torch.nn as nn
import torch.nn.functional as F

# Configuration / default parameters
batch_size = 8
in_channels = 16
depth = 8
height = 8
width = 8

# Model hyperparameters (also returned by get_init_inputs)
OUT_CHANNELS_1 = 64
OUT_CHANNELS_2 = 32
STYLE_DIM = 128
KERNEL1 = (4, 4, 4)
STRIDE1 = (2, 2, 2)
PADDING1 = 1
KERNEL2 = (3, 3, 3)
STRIDE2 = 1
PADDING2 = 1
DROPOUT_P = 0.12


class Model(nn.Module):
    """
    3D upsampling & style-modulated block using LazyConvTranspose3d,
    InstanceNorm3d and FeatureAlphaDropout. The module demonstrates:
      - lazy transposed convolutions to upsample spatial dims
      - instance normalization followed by learned style modulation
      - non-linearity and FeatureAlphaDropout for regularization
      - a second transposed convolution and global pooling for outputs

    Forward signature:
      forward(x: Tensor, style: Tensor) -> Tensor
    where:
      x: (B, C_in, D, H, W)
      style: (B, style_dim)
    """
    def __init__(
        self,
        out_channels1: int = OUT_CHANNELS_1,
        out_channels2: int = OUT_CHANNELS_2,
        style_dim: int = STYLE_DIM,
        dropout_p: float = DROPOUT_P,
        kernel1=KERNEL1,
        stride1=STRIDE1,
        padding1=PADDING1,
        kernel2=KERNEL2,
        stride2=STRIDE2,
        padding2=PADDING2,
    ):
        super(Model, self).__init__()

        # First upsampling block (lazy in_channels determined on first forward)
        self.up1 = nn.LazyConvTranspose3d(
            out_channels=out_channels1,
            kernel_size=kernel1,
            stride=stride1,
            padding=padding1,
            bias=False,
        )

        # InstanceNorm expects the exact number of channels produced by up1
        self.inst_norm = nn.InstanceNorm3d(num_features=out_channels1, affine=False)

        # Style projection will produce per-channel scale and shift (gamma, beta)
        # Shape -> (B, out_channels1 * 2), split into gamma and beta
        self.style_proj = nn.Linear(style_dim, out_channels1 * 2)

        # Non-linearity and dropout
        self.act = nn.GELU()
        self.dropout = nn.FeatureAlphaDropout(p=dropout_p)

        # Second transposed conv to refine channels/spatial layout
        self.up2 = nn.LazyConvTranspose3d(
            out_channels=out_channels2,
            kernel_size=kernel2,
            stride=stride2,
            padding=padding2,
            bias=True,
        )

        # Final small projection (1x1x1 conv simulated with ConvTranspose3d kernel=1)
        # Use standard ConvTranspose3d with kernel_size=1 to act as pointwise conv (lazy)
        self.final_proj = nn.LazyConvTranspose3d(out_channels=out_channels2, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the upsampling + style modulation pipeline.

        Steps:
          1. Upsample spatial dims with up1 (LazyConvTranspose3d)
          2. Instance normalization
          3. Compute style gamma/beta and apply modulation: out = normed * (1 + gamma) + beta
          4. Activation (GELU) and FeatureAlphaDropout
          5. Second transposed conv (up2) for refinement
          6. Final projection (pointwise) and global spatial pooling to produce compact vector

        Args:
            x: Input tensor with shape (B, C_in, D, H, W)
            style: Style tensor with shape (B, style_dim)

        Returns:
            Tensor of shape (B, out_channels2) after global pooling
        """
        # 1) Upsample
        out = self.up1(x)  # -> (B, out_channels1, D*2, H*2, W*2)

        # 2) Instance normalization (per-sample, per-channel)
        out = self.inst_norm(out)

        # 3) Style modulation
        # style_proj -> (B, 2 * out_channels1) -> split -> reshape to broadcast
        style_params = self.style_proj(style)  # (B, 2*C)
        C = style_params.shape[1] // 2
        gamma, beta = style_params[:, :C], style_params[:, C:]
        # reshape to (B, C, 1, 1, 1) for spatial broadcast
        gamma = gamma.view(-1, C, 1, 1, 1)
        beta = beta.view(-1, C, 1, 1, 1)
        # apply modulation; adding 1 to gamma implements residual scaling
        out = out * (1.0 + gamma) + beta

        # 4) Activation and dropout
        out = self.act(out)
        out = self.dropout(out)

        # 5) Second transposed conv (refinement)
        out = self.up2(out)  # -> (B, out_channels2, D*2*stride2, H*2*stride2, W*2*stride2)

        # 6) Final projection and global pooling
        out = self.final_proj(out)  # pointwise refinement
        # Global average pooling over spatial dims -> (B, out_channels2)
        out = out.mean(dim=[2, 3, 4])

        return out


def get_inputs():
    """
    Returns:
      - x: 5D tensor (batch_size, in_channels, D, H, W)
      - style: 2D tensor (batch_size, STYLE_DIM)
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    style = torch.randn(batch_size, STYLE_DIM)
    return [x, style]


def get_init_inputs():
    """
    Returns the initialization arguments for Model:
      [out_channels1, out_channels2, style_dim, dropout_p, kernel1, stride1, padding1, kernel2, stride2, padding2]
    """
    return [
        OUT_CHANNELS_1,
        OUT_CHANNELS_2,
        STYLE_DIM,
        DROPOUT_P,
        KERNEL1,
        STRIDE1,
        PADDING1,
        KERNEL2,
        STRIDE2,
        PADDING2,
    ]