import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Sequence feature compressor and reconstructer.

    This module:
    - Performs adaptive 1D average pooling across the temporal dimension to compress sequence length.
    - Applies a two-layer pointwise MLP (Linear -> Tanhshrink -> Linear) on pooled features.
    - Upsamples the processed features back to the original sequence length using linear interpolation.
    - Adds a residual connection from the original input and applies a final Tanhshrink.

    Input shape: (batch, seq_len, in_features)
    Output shape: (batch, seq_len, in_features)
    """
    def __init__(self, in_features: int, hidden_dim: int, pooled_len: int):
        super(Model, self).__init__()
        # Pointwise MLP applied to pooled time-steps
        self.linear1 = nn.Linear(in_features, hidden_dim, bias=True)
        self.activation = nn.Tanhshrink()
        self.linear2 = nn.Linear(hidden_dim, in_features, bias=True)
        # Adaptive pooling to compress temporal dimension to pooled_len
        self.pool = nn.AdaptiveAvgPool1d(pooled_len)
        self.pooled_len = pooled_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, T, F) where
               B = batch size, T = sequence length, F = in_features

        Returns:
            Tensor of same shape (B, T, F) after compress->transform->reconstruct.
        """
        # x: (B, T, F) -> (B, F, T) for AdaptiveAvgPool1d
        x_perm = x.permute(0, 2, 1)
        # Adaptive average pool along temporal dimension: (B, F, pooled_len)
        pooled = self.pool(x_perm)
        # Back to time-major for pointwise linear layers: (B, pooled_len, F)
        pooled_t = pooled.permute(0, 2, 1)
        # Pointwise MLP across feature dimension applied independently per time-step
        hidden = self.linear1(pooled_t)          # (B, pooled_len, hidden_dim)
        hidden = self.activation(hidden)         # element-wise nonlinearity
        processed = self.linear2(hidden)         # (B, pooled_len, F)
        # Prepare for interpolation: (B, F, pooled_len)
        processed_perm = processed.permute(0, 2, 1)
        # Upsample back to original temporal resolution using linear interpolation
        # F.interpolate expects (B, C, L)
        upsampled = F.interpolate(processed_perm, size=x.shape[1], mode='linear', align_corners=True)
        # (B, F, T) -> (B, T, F)
        upsampled_t = upsampled.permute(0, 2, 1)
        # Residual connection with original input and final activation
        out = self.activation(x + upsampled_t)
        return out

# Configuration / default sizes
batch_size = 8
seq_len = 512
in_features = 1024
hidden_dim = 2048
pooled_len = 64

def get_inputs():
    """
    Returns:
        A list with a single input tensor of shape (batch_size, seq_len, in_features)
    """
    x = torch.randn(batch_size, seq_len, in_features)
    return [x]

def get_init_inputs():
    """
    Returns initialization arguments for the Model constructor.
    """
    return [in_features, hidden_dim, pooled_len]