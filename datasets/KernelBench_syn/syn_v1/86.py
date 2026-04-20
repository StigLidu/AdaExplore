import torch
import torch.nn as nn

# Configuration / tensor sizes
BATCH = 8
C_IN = 64
C_OUT = 128
H = 32
W = 32
P_H = 8
P_W = 8
K = 512  # final feature dimension after spatial projection

class Model(nn.Module):
    """
    Complex feature extractor that:
      - Normalizes per-instance across channels
      - Applies a randomized leaky activation (RReLU)
      - Performs learned channel mixing via an einsum-projection
      - Spatially compresses features with AdaptiveMaxPool2d
      - Projects spatial locations to a K-dimensional embedding using an external projection matrix
      - Applies per-channel modulation and reduces to a final (batch, K) feature

    Inputs:
      X: Tensor of shape (BATCH, C_IN, H, W)
      spatial_proj: Tensor of shape (P_H * P_W, K) (external projection for pooled spatial locations)
      channel_mod: Tensor of shape (BATCH, C_OUT) (per-batch per-channel modulation factors)

    Output:
      Tensor of shape (BATCH, K)
    """
    def __init__(self):
        super(Model, self).__init__()
        # Instance normalization across channels (per-sample normalization)
        self.inst_norm = nn.InstanceNorm2d(C_IN, affine=False, track_running_stats=False)

        # Randomized leaky ReLU activation (stochastic in training, deterministic in eval)
        self.activation = nn.RReLU(lower=0.125, upper=0.333, inplace=False)

        # Adaptive max pooling to reduce spatial resolution to (P_H, P_W)
        self.pool = nn.AdaptiveMaxPool2d((P_H, P_W))

        # Learned channel projection matrix (C_IN -> C_OUT)
        # Registered as a parameter so it can be optimized.
        self.channel_proj = nn.Parameter(torch.randn(C_IN, C_OUT) * (1.0 / (C_IN ** 0.5)))

        # Small epsilon for numerical stability
        self.eps = 1e-6

    def forward(self, X, spatial_proj, channel_mod):
        """
        Forward pass.

        Args:
            X (torch.Tensor): Input feature map, shape (BATCH, C_IN, H, W)
            spatial_proj (torch.Tensor): Spatial projection matrix, shape (P_H*P_W, K)
            channel_mod (torch.Tensor): Per-batch per-channel modulation, shape (BATCH, C_OUT)

        Returns:
            torch.Tensor: Output tensor of shape (BATCH, K)
        """
        # 1) Instance normalization (per-sample, per-channel)
        x = self.inst_norm(X)  # (B, C_IN, H, W)

        # 2) Non-linear activation (RReLU)
        x = self.activation(x)  # (B, C_IN, H, W)

        # 3) Channel mixing via einsum using learned projection
        #    from (B, C_IN, H, W) and (C_IN, C_OUT) -> (B, C_OUT, H, W)
        x = torch.einsum("bchw,co->bohw", x, self.channel_proj)  # (B, C_OUT, H, W)

        # 4) Spatial compression with adaptive max pooling
        x = self.pool(x)  # (B, C_OUT, P_H, P_W)

        # 5) Flatten spatial dims to (P = P_H * P_W)
        B, C, pH, pW = x.shape
        x = x.reshape(B, C, pH * pW)  # (B, C_OUT, P)

        # 6) Project spatial locations into K-dim using external matrix spatial_proj (P, K)
        #    Use einsum to compute per-channel per-batch projections: (B, C, P) x (P, K) -> (B, C, K)
        z = torch.einsum("bcp,pk->bck", x, spatial_proj)  # (B, C_OUT, K)

        # 7) Apply channel-wise modulation (per-batch) and aggregate channels
        #    channel_mod has shape (B, C_OUT), expand to match z (B, C_OUT, K)
        mod = channel_mod.unsqueeze(-1)  # (B, C_OUT, 1)
        z = z * mod  # (B, C_OUT, K)

        # 8) Sum across channels to obtain final (B, K) representation
        out = z.sum(dim=1)  # (B, K)

        # 9) Optional l2-normalize the output for stability
        norm = out.norm(p=2, dim=1, keepdim=True).clamp(min=self.eps)
        out = out / norm

        return out

# Public factory functions required by the harness

def get_inputs():
    """
    Returns:
      [X, spatial_proj, channel_mod]
        X: random tensor (BATCH, C_IN, H, W)
        spatial_proj: random tensor (P_H * P_W, K)
        channel_mod: random tensor (BATCH, C_OUT)
    """
    X = torch.randn(BATCH, C_IN, H, W)
    spatial_proj = torch.randn(P_H * P_W, K)
    channel_mod = torch.randn(BATCH, C_OUT)
    return [X, spatial_proj, channel_mod]

def get_init_inputs():
    """
    No special initialization required for constructing Model() in this example.
    """
    return []