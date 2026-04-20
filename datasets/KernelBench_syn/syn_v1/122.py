import torch
import torch.nn as nn
import torch.nn.functional as F

# Configuration / default sizes
BATCH_SIZE = 8
SEQ_LEN = 1024
IN_CHANNELS = 64
POOL_OUTPUT_SIZE = 16
HIDDEN_DIM = 128
OUT_FEATURES = 256
RMS_EPS = 1e-6

class Model(nn.Module):
    """
    A moderately complex sequence processing module that demonstrates a mix of
    pooling, normalization, linear projections and batch normalization using:
      - nn.AdaptiveMaxPool1d
      - nn.RMSNorm
      - nn.SyncBatchNorm

    Computation pipeline (high level):
      1. Expect input x of shape (batch, seq_len, in_channels).
      2. Permute to (batch, in_channels, seq_len) to apply AdaptiveMaxPool1d,
         reducing the sequence length to `pool_output_size`.
      3. Permute to (batch, pool_output_size, in_channels) and apply RMSNorm
         over the channel dimension for each position.
      4. Apply a position-wise Linear(in_channels -> hidden_dim) and ReLU.
      5. Permute to (batch, hidden_dim, pool_output_size) and apply SyncBatchNorm
         across the hidden channels.
      6. Permute back, flatten positional and channel dims, and apply a final
         Linear to produce an output of shape (batch, out_features).

    This arrangement tests channel-positional reshaping, different normalization
    semantics (RMS over last-dim vs SyncBatchNorm over channel-dim), and pooling.
    """
    def __init__(
        self,
        in_channels: int = IN_CHANNELS,
        hidden_dim: int = HIDDEN_DIM,
        pool_output_size: int = POOL_OUTPUT_SIZE,
        out_features: int = OUT_FEATURES,
        rms_eps: float = RMS_EPS,
    ):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.pool_output_size = pool_output_size
        self.out_features = out_features

        # Adaptive max pooling over the sequence dimension
        self.pool = nn.AdaptiveMaxPool1d(output_size=self.pool_output_size)

        # RMSNorm applied to the last dimension (channels) after we permute to (B, P, C)
        self.rmsnorm = nn.RMSNorm(normalized_shape=self.in_channels, eps=rms_eps)

        # Position-wise linear layer mapping channels -> hidden_dim
        self.position_linear = nn.Linear(self.in_channels, self.hidden_dim, bias=True)

        # SyncBatchNorm operates over the hidden channel dimension (C dimension is second)
        # We'll feed it tensors shaped (B, C, L)
        self.sync_bn = nn.SyncBatchNorm(num_features=self.hidden_dim)

        # Final linear layer mapping flattened (P * hidden_dim) -> out_features
        self.final_linear = nn.Linear(self.hidden_dim * self.pool_output_size, self.out_features, bias=True)

        # Small projection vector to optionally re-scale pooled features (learned)
        self.post_scale = nn.Parameter(torch.ones(self.hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch, seq_len, in_channels)

        Returns:
            Tensor of shape (batch, out_features)
        """
        # Validate shape
        if x.dim() != 3 or x.size(2) != self.in_channels:
            raise ValueError(f"Expected input shape (B, seq_len, {self.in_channels}), got {tuple(x.shape)}")

        # 1) Permute to (B, C, L) for pooling
        x = x.permute(0, 2, 1)  # (B, C, L)

        # 2) Adaptive max pool along the sequence length -> (B, C, P)
        x = self.pool(x)  # (B, C, P)

        # 3) Permute to (B, P, C) for RMSNorm across channels per position
        x = x.permute(0, 2, 1)  # (B, P, C)

        # 4) RMS normalization over channels
        x = self.rmsnorm(x)  # (B, P, C)

        # 5) Position-wise linear: apply to last dim (channels -> hidden_dim)
        # Flatten positions into batch for efficiency or use linear directly on last dim
        B, P, C = x.shape
        x = self.position_linear(x)  # (B, P, hidden_dim)

        # 6) Non-linearity
        x = F.relu(x, inplace=False)

        # 7) Optional learned scaling per hidden channel (applied across positions)
        x = x * self.post_scale.view(1, 1, -1)  # (B, P, hidden_dim)

        # 8) Prepare for SyncBatchNorm which expects (B, C, L)
        x = x.permute(0, 2, 1)  # (B, hidden_dim, P)

        # 9) Synchronized BatchNorm across processes/devices (or local batch if single)
        x = self.sync_bn(x)  # (B, hidden_dim, P)

        # 10) Permute back to (B, P, hidden_dim) and flatten for final projection
        x = x.permute(0, 2, 1)  # (B, P, hidden_dim)
        x = x.reshape(B, P * self.hidden_dim)  # (B, P * hidden_dim)

        # 11) Final linear projection to out_features
        out = self.final_linear(x)  # (B, out_features)

        return out

# Module-level get_inputs and get_init_inputs to match example structure

def get_inputs():
    """
    Returns a list containing a single input tensor appropriate for the Model:
      - Tensor shape: (BATCH_SIZE, SEQ_LEN, IN_CHANNELS)
    """
    x = torch.randn(BATCH_SIZE, SEQ_LEN, IN_CHANNELS)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for the Model constructor in order:
      (in_channels, hidden_dim, pool_output_size, out_features, rms_eps)
    """
    return [IN_CHANNELS, HIDDEN_DIM, POOL_OUTPUT_SIZE, OUT_FEATURES, RMS_EPS]