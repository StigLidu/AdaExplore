import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex patch-based feature extractor that demonstrates a combination of:
      - nn.Unfold to extract sliding local patches,
      - a learnable linear projection across patch elements,
      - nn.SELU nonlinearity,
      - nn.LPPool1d to perform structured pooling along a spatial dimension,
      - and a learned channel gating derived from the original input.

    Forward signature:
        forward(x: torch.Tensor) -> torch.Tensor
    where x has shape (N, C_in, H, W) and the output is (N, C_out).
    """
    def __init__(self):
        super(Model, self).__init__()

        # Configuration pulled from the module-level constants
        self.in_channels = IN_C
        self.patch_k = K
        self.out_features = OUT_FEAT
        self.pool_k = POOL_K
        self.lp_p = LP_P
        self.H = H
        self.W = W

        # Unfold to extract sliding patches (preserve spatial dims via padding)
        self.unfold = nn.Unfold(kernel_size=self.patch_k, padding=self.patch_k // 2, stride=1)

        # SELU activation
        self.selu = nn.SELU()

        # LPPool1d to pool along the width dimension after rearrangement
        # LPPool1d expects (N, C, L) so we'll craft that shape in forward
        self.lppool = nn.LPPool1d(norm_type=self.lp_p, kernel_size=self.pool_k, stride=self.pool_k)

        # Learnable linear projection from patch dimension (C * K * K) -> OUT_FEAT
        patch_dim = self.in_channels * (self.patch_k * self.patch_k)
        self.weight = nn.Parameter(torch.randn(patch_dim, self.out_features) * (1.0 / patch_dim**0.5))
        self.bias = nn.Parameter(torch.zeros(self.out_features))

        # Learnable projection for channel gating: from original input channels to OUT_FEAT
        self.scale_proj = nn.Parameter(torch.randn(self.in_channels, self.out_features) * 0.1)
        self.scale_bias = nn.Parameter(torch.zeros(self.out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input images of shape (N, C_in, H, W)

        Returns:
            torch.Tensor: Feature tensor of shape (N, OUT_FEAT)
        """
        N, C, H, W = x.shape
        assert C == self.in_channels and H == self.H and W == self.W, \
            f"Expected input shape (N, {self.in_channels}, {self.H}, {self.W}), got {x.shape}"

        # 1) Extract patches: out_unf shape -> (N, C * K * K, L) where L = H * W (with padding and stride=1)
        patches = self.unfold(x)  # (N, patch_dim, L)

        # 2) Move patch dimension to last for a linear projection: (N, L, patch_dim)
        patches = patches.permute(0, 2, 1)

        # 3) Linear projection per patch: (N, L, OUT_FEAT)
        projected = torch.matmul(patches, self.weight) + self.bias  # (N, L, OUT_FEAT)

        # 4) Reshape to spatial feature map: (N, OUT_FEAT, H, W)
        projected = projected.permute(0, 2, 1).contiguous().view(N, self.out_features, H, W)

        # 5) Apply SELU nonlinearity
        activated = self.selu(projected)

        # 6) Prepare for 1D pooling along width:
        #    combine channel and height to form new 'channels' dimension so pooling acts along W
        c_comb = self.out_features * H
        x_for_pool = activated.view(N, c_comb, W)  # (N, C_comb, W)

        # 7) Apply LPPool1d along the width dimension
        pooled = self.lppool(x_for_pool)  # (N, C_comb, W_p)
        W_p = pooled.shape[-1]

        # 8) Restore spatial layout: (N, OUT_FEAT, H, W_p)
        pooled = pooled.view(N, self.out_features, H, W_p)

        # 9) Compute a channel gating vector from the original input by global averaging over spatial dims
        #    (N, C_in)
        channel_summary = x.mean(dim=(2, 3))

        # 10) Project the channel summary into gating values for OUT_FEAT and squash with sigmoid
        gate = torch.matmul(channel_summary, self.scale_proj) + self.scale_bias  # (N, OUT_FEAT)
        gate = torch.sigmoid(gate).unsqueeze(-1).unsqueeze(-1)  # (N, OUT_FEAT, 1, 1)

        # 11) Apply gating to the pooled features
        gated = pooled * gate  # (N, OUT_FEAT, H, W_p)

        # 12) Global average across remaining spatial dims -> final feature vector (N, OUT_FEAT)
        out = gated.mean(dim=(2, 3))

        return out

# Module-level configuration variables
BATCH = 8
IN_C = 3
H = 32
W = 32
K = 3           # patch kernel size for Unfold
OUT_FEAT = 64   # output feature dimension after projection
POOL_K = 2      # kernel/stride for LPPool1d
LP_P = 2        # L_p norm for LPPool1d

def get_inputs():
    """
    Returns a list containing a single input tensor for the model:
    - x: random image tensor of shape (BATCH, IN_C, H, W)
    """
    x = torch.randn(BATCH, IN_C, H, W)
    return [x]

def get_init_inputs():
    """
    No special runtime initialization inputs; model is self-contained.
    """
    return []