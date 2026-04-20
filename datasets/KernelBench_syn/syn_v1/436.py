import torch
import torch.nn as nn

# Configuration (module-level)
BATCH = 4         # batch size
IN_CHANNELS = 3   # input channels (e.g., RGB)
DEPTH = 16        # depth dimension for 3D volume
HEIGHT = 64       # height of spatial dims
WIDTH = 64        # width of spatial dims

OUT_CHANNELS = 32  # number of output channels for Conv3d
CONV_KERNEL = (3, 5, 5)
POOL_KERNEL = 2

class Model(nn.Module):
    """
    Complex model combining a lazy 3D convolution, sigmoid gating, 2D average pooling,
    and a learned channel projection to produce a depth-pooled feature vector.

    Computation summary:
      1. Apply LazyConv3d to input X: (B, C, D, H, W) -> (B, outC, D', H', W')
      2. Compute elementwise sigmoid gate and multiply with conv output.
      3. Rearrange and merge batch+depth to apply AvgPool2d over spatial dims.
      4. Collapse spatial dims to obtain per-(batch,depth,channel) summaries.
      5. Compute per-depth scores via a learned channel projection, scaled by alpha and sigmoid.
      6. Weight and sum summaries across depth to produce final (B, outC) output.
    """
    def __init__(self, out_channels: int = OUT_CHANNELS):
        super(Model, self).__init__()
        # LazyConv3d will infer in_channels on the first forward pass
        self.conv3d = nn.LazyConv3d(out_channels=out_channels, kernel_size=CONV_KERNEL, padding=(1,2,2))
        # 2D average pooling to reduce spatial HxW after merging depth into batch
        self.pool2d = nn.AvgPool2d(kernel_size=POOL_KERNEL)
        # Elementwise non-linearity used as gating
        self.sigmoid = nn.Sigmoid()
        # Learnable projection vector over channels to compute depth-attention scores.
        # Size matches out_channels (known parameter).
        self.channel_proj = nn.Parameter(torch.randn(out_channels))

    def forward(self, X: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X: Input tensor of shape (B, C_in, D, H, W)
            alpha: Scalar factor to scale attention logits before sigmoid

        Returns:
            Tensor of shape (B, out_channels): depth-pooled channel features
        """
        # 1) 3D convolution -> shape: (B, outC, D1, H1, W1)
        conv_out = self.conv3d(X)

        # 2) Sigmoid gating applied elementwise and multiply
        gate = self.sigmoid(conv_out)
        gated = conv_out * gate  # elementwise gated features

        # 3) Move depth dimension next to batch and merge them to apply 2D pooling
        # gated: (B, outC, D1, H1, W1) -> permute -> (B, D1, outC, H1, W1)
        b, outc, d1, h1, w1 = gated.shape
        merged = gated.permute(0, 2, 1, 3, 4).reshape(b * d1, outc, h1, w1)

        # 4) Apply 2D average pooling over spatial dims -> (B*d1, outC, H2, W2)
        pooled = self.pool2d(merged)

        # 5) Reduce spatial dims by mean to obtain per-(batch*depth, channel)
        spatial_summary = pooled.mean(dim=[2, 3])  # shape: (B*d1, outC)

        # 6) Restore (B, D1, outC)
        summary = spatial_summary.view(b, d1, outc)  # (B, D1, outC)

        # 7) Compute per-depth attention scores:
        #    Use einsum to combine channel features with learned channel projection -> (B, D1)
        #    Scale by alpha, then pass through sigmoid to obtain positive weights in (0,1)
        raw_scores = torch.einsum("b d o, o -> b d", summary, self.channel_proj)
        attn = self.sigmoid(raw_scores * alpha)  # (B, D1)

        # 8) Normalize attention across depth to form a convex combination
        attn_sum = attn.sum(dim=1, keepdim=True) + 1e-6
        attn_norm = attn / attn_sum  # (B, D1)

        # 9) Weighted sum across depth to get final features: (B, outC)
        out = torch.einsum("b d, b d o -> b o", attn_norm, summary)

        return out

# Default sizes for get_inputs
B = BATCH
C = IN_CHANNELS
D = DEPTH
H = HEIGHT
W = WIDTH

def get_inputs():
    """
    Returns sample inputs to run the model.
    X: random 5D tensor of shape (B, C, D, H, W)
    alpha: scalar float controlling attention sharpness
    """
    X = torch.randn(B, C, D, H, W)
    alpha = 2.0  # moderately sharp attention
    return [X, alpha]

def get_init_inputs():
    """
    No special initialization inputs required; LazyConv3d will lazily initialize
    its in_channels on the first forward pass.
    """
    return []