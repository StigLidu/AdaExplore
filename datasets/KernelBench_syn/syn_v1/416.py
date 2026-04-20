import torch
import torch.nn as nn

"""
Complex model that mixes temporal and channel information with normalization and gating.

Computation graph:
    1. X_norm = InstanceNorm1d(X)                    # normalize per-instance per-channel across time
    2. T = X_norm @ W_time                            # linear mix across time dimension
    3. C = permute(T) @ W_channel                     # linear mix across channels (performed on permuted tensor)
    4. G = C * sigmoid(C)                             # gating implemented via LogSigmoid for numerical stability
       (sigmoid(C) = exp(LogSigmoid(C)))
    5. G_ln = LayerNorm(G_permuted)                   # layer norm across channel dim
    6. out = G_ln_permuted + X_norm                   # residual connection back to original shape

Inputs:
    X: (B, C, T)         - main input tensor
    W_time: (T, T)       - temporal mixing matrix
    W_channel: (C, C)    - channel mixing matrix
"""

# Configuration
BATCH = 8
CHANNELS = 64
SEQ_LEN = 128
EPS = 1e-5
AFFINE = True

class Model(nn.Module):
    def __init__(self, channels: int, seq_len: int, eps: float = EPS, affine: bool = AFFINE):
        """
        Args:
            channels (int): number of channels C
            seq_len (int): temporal length T
            eps (float): epsilon for InstanceNorm1d
            affine (bool): whether InstanceNorm1d has affine parameters
        """
        super(Model, self).__init__()
        # Normalizes each instance per channel across the temporal dimension
        self.inst_norm = nn.InstanceNorm1d(num_features=channels, eps=eps, affine=affine)
        # LayerNorm will be applied to tensors of shape (B, T, C), normalizing over last dim (channels)
        self.layer_norm = nn.LayerNorm(normalized_shape=channels, eps=1e-5)
        # LogSigmoid used to build a numerically-stable sigmoid for gating
        self.logsig = nn.LogSigmoid()
        # store shapes for potential checks; not used in computation
        self.channels = channels
        self.seq_len = seq_len

    def forward(self, X: torch.Tensor, W_time: torch.Tensor, W_channel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X (torch.Tensor): Input tensor of shape (B, C, T)
            W_time (torch.Tensor): Temporal mixing matrix of shape (T, T)
            W_channel (torch.Tensor): Channel mixing matrix of shape (C, C)

        Returns:
            torch.Tensor: Output tensor of shape (B, C, T)
        """
        # 1) Instance normalization across the temporal axis for each (B, C)
        X_norm = self.inst_norm(X)  # (B, C, T)

        # 2) Temporal mixing: mix each channel's time series via W_time
        #    (B, C, T) @ (T, T) -> (B, C, T)
        T_mixed = torch.matmul(X_norm, W_time)

        # 3) Channel mixing: permute to (B, T, C) to apply channel matrix on last dim
        T_permuted = T_mixed.permute(0, 2, 1)  # (B, T, C)
        C_mixed = torch.matmul(T_permuted, W_channel)  # (B, T, C)

        # 4) Gating: use LogSigmoid to compute sigmoid(C_mixed) as exp(LogSigmoid(C_mixed))
        sigmoid_vals = torch.exp(self.logsig(C_mixed))
        gated = C_mixed * sigmoid_vals  # (B, T, C)

        # 5) Layer normalization across channels (last dim of (B, T, C))
        gated_ln = self.layer_norm(gated)  # (B, T, C)

        # 6) Residual connection back to (B, C, T)
        out = gated_ln.permute(0, 2, 1) + X_norm  # (B, C, T)

        return out

def get_inputs():
    """
    Returns:
        [X, W_time, W_channel]
        X: (BATCH, CHANNELS, SEQ_LEN)
        W_time: (SEQ_LEN, SEQ_LEN)
        W_channel: (CHANNELS, CHANNELS)
    """
    X = torch.randn(BATCH, CHANNELS, SEQ_LEN)
    W_time = torch.randn(SEQ_LEN, SEQ_LEN)
    W_channel = torch.randn(CHANNELS, CHANNELS)
    return [X, W_time, W_channel]

def get_init_inputs():
    """
    Returns the initialization parameters for Model: [channels, seq_len, eps, affine]
    """
    return [CHANNELS, SEQ_LEN, EPS, AFFINE]