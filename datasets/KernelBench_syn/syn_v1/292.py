import torch
import torch.nn as nn

# Configuration variables
batch_size = 8
in_channels = 48
depth = 16
height = 32
width = 32

# Model hyperparameters (used for initialization)
mid_channels = 24           # bottleneck channel size for channel-wise projection
dropout_p = 0.1             # dropout probability for AlphaDropout
softplus_beta = 1.0         # beta parameter for Softplus
softplus_threshold = 20.0   # threshold parameter for Softplus

class Model(nn.Module):
    """
    Complex 3D feature gating module combining LazyBatchNorm3d, Softplus activation,
    and AlphaDropout to compute an input-dependent channel gating and residual fusion.

    Computation steps:
      1. Apply Lazy BatchNorm3d to stabilize channel statistics.
      2. Global-spatial average to produce channel descriptors.
      3. Two-layer channel projection (fc -> Softplus -> AlphaDropout -> fc) to produce gates.
      4. Sigmoid gating applied per-channel and broadcasted over spatial dims.
      5. Residual fusion between gated-normalized features and a scaled original input.
      6. Final Softplus nonlinearity to produce output.

    Inputs:
      x: Tensor of shape (batch_size, in_channels, depth, height, width)

    Returns:
      Tensor of same shape as input with per-channel modulation applied.
    """
    def __init__(self, in_ch: int, mid_ch: int, dropout_p: float = 0.1,
                 softplus_beta: float = 1.0, softplus_threshold: float = 20.0):
        super(Model, self).__init__()
        # LazyBatchNorm3d will infer num_features on first forward pass
        self.bn = nn.LazyBatchNorm3d()
        # Channel projection: reduce -> expand back to produce channel-wise gating
        self.fc_reduce = nn.Linear(in_ch, mid_ch)
        self.fc_expand = nn.Linear(mid_ch, in_ch)
        # Activation and regularization
        self.softplus = nn.Softplus(beta=softplus_beta, threshold=softplus_threshold)
        self.alpha_dropout = nn.AlphaDropout(p=dropout_p)
        # A small learnable scalar for residual scaling (initialized small)
        self.res_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the gating module.

        Args:
            x (torch.Tensor): Input tensor with shape (B, C, D, H, W)

        Returns:
            torch.Tensor: Output tensor with same shape as input.
        """
        # 1) Normalize
        x_bn = self.bn(x)  # (B, C, D, H, W)

        # 2) Global-spatial descriptor
        # Aggregate across depth, height, width -> (B, C)
        desc = x_bn.mean(dim=(2, 3, 4))  # (B, C)

        # 3) Channel projection and nonlinearity
        z = self.fc_reduce(desc)         # (B, mid_ch)
        z = self.softplus(z)             # Softplus activation
        z = self.alpha_dropout(z)        # AlphaDropout for self-normalizing networks

        # 4) Expand back to channel dimension and produce gating logits
        logits = self.fc_expand(z)       # (B, C)
        gates = torch.sigmoid(logits)    # (B, C) in (0,1)

        # 5) Reshape gates and apply to normalized features (channel-wise)
        gates = gates.view(gates.size(0), gates.size(1), 1, 1, 1)  # (B, C, 1, 1, 1)
        gated = x_bn * gates  # (B, C, D, H, W)

        # 6) Residual fusion with a scaled original input and final nonlinearity
        out = gated + self.res_scale * x
        out = self.softplus(out)  # final activation

        return out

# Test input configuration (aligns with module-level variables)
def get_inputs():
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    # Return initialization parameters for the Model constructor:
    # in_channels, mid_channels, dropout_p, softplus_beta, softplus_threshold
    return [in_channels, mid_channels, dropout_p, softplus_beta, softplus_threshold]