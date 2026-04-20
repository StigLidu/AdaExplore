import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class Model(nn.Module):
    """
    1D temporal processing block that combines Zero padding, Lp pooling,
    Alpha Dropout, pointwise projection, a Swish-like activation, and a
    residual path with LayerNorm.

    Forward computation:
        1. Zero-pad the input along the temporal dimension.
        2. Apply Lp pooling (LPPool1d) to aggregate local neighborhoods.
        3. Apply AlphaDropout for regularization.
        4. Project channel dimension with a 1x1 Conv1d.
        5. Apply Swish activation (x * sigmoid(x)).
        6. Create a residual by adaptive-averaging the original (unpadded)
           input to the pooled temporal length and projecting to out_features.
        7. Sum projection + residual and apply LayerNorm across channels.

    Input:
        x: Tensor of shape (batch_size, in_channels, length)

    Output:
        Tensor of shape (batch_size, out_features, pooled_length)
    """
    def __init__(
        self,
        in_channels: int,
        out_features: int,
        lp_norm: int,
        pool_kernel: int,
        pool_stride: int,
        pad_left: int,
        pad_right: int,
        dropout_p: float
    ):
        super(Model, self).__init__()
        # Zero padding on temporal dimension
        self.pad = nn.ZeroPad1d((pad_left, pad_right))
        # Lp pooling (1D)
        self.lp_pool = nn.LPPool1d(lp_norm, kernel_size=pool_kernel, stride=pool_stride)
        # Alpha Dropout for self-normalizing networks compatibility
        self.alpha_dropout = nn.AlphaDropout(p=dropout_p)
        # Pointwise projection (1x1 conv) after pooling to change channel dimension
        self.proj = nn.Conv1d(in_channels, out_features, kernel_size=1, bias=True)
        # Residual projection (1x1 conv) on downsampled original input
        self.res_proj = nn.Conv1d(in_channels, out_features, kernel_size=1, bias=True)
        # LayerNorm will be applied on [batch, length, channels] shape
        self.norm = nn.LayerNorm(normalized_shape=out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, L)

        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, L_pooled)
        """
        # 1) Pad temporal boundaries
        x_padded = self.pad(x)  # (B, C_in, L + pad_left + pad_right)

        # 2) Lp pooling reduces temporal resolution
        pooled = self.lp_pool(x_padded)  # (B, C_in, L_pooled)

        # 3) Alpha Dropout for robust regularization
        dropped = self.alpha_dropout(pooled)  # (B, C_in, L_pooled)

        # 4) Pointwise projection to out_features
        projected = self.proj(dropped)  # (B, C_out, L_pooled)

        # 5) Swish activation (elementwise)
        activated = projected * torch.sigmoid(projected)  # (B, C_out, L_pooled)

        # 6) Residual path: downsample original (unpadded) input to pooled length
        target_len = activated.size(2)
        residual_down = F.adaptive_avg_pool1d(x, output_size=target_len)  # (B, C_in, L_pooled)
        residual = self.res_proj(residual_down)  # (B, C_out, L_pooled)

        # 7) Add residual and apply LayerNorm across channels
        out = activated + residual  # (B, C_out, L_pooled)
        # LayerNorm expects (B, L, C), so transpose, normalize, transpose back
        out = out.transpose(1, 2)  # (B, L_pooled, C_out)
        out = self.norm(out)
        out = out.transpose(1, 2)  # (B, C_out, L_pooled)

        return out

# Module-level configuration variables
batch_size = 8
in_channels = 64
length = 1024
out_features = 128

lp_norm = 2             # L2 pooling
pool_kernel = 5
pool_stride = 2
pad_left = 2
pad_right = 2
dropout_p = 0.1

def get_inputs() -> List[torch.Tensor]:
    """
    Returns the random input tensor for testing the Model.

    Shape:
        x: (batch_size, in_channels, length)
    """
    x = torch.randn(batch_size, in_channels, length)
    return [x]

def get_init_inputs() -> List:
    """
    Returns the initialization parameters for Model in the same order
    as Model.__init__'s arguments (excluding self).
    """
    return [
        in_channels,
        out_features,
        lp_norm,
        pool_kernel,
        pool_stride,
        pad_left,
        pad_right,
        dropout_p
    ]