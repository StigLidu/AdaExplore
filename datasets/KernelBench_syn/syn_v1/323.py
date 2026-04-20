import torch
import torch.nn as nn

# Configuration variables
batch_size = 8
channels = 3
depth = 8
height = 16
width = 16

# Pooling / activation configuration
lp_norm_type = 2          # p for LPPool2d
avgpool_kernel = (2, 2, 2)
lppool_kernel = 3
eps = 1e-6


class Model(nn.Module):
    """
    A composite module that demonstrates a multi-stage spatial reduction and gating pipeline:
      1. 3D average pooling to reduce depth/height/width.
      2. Per-sample per-channel L2 normalization across spatial dims.
      3. Collapse depth into the channel dimension and apply an Lp (2-norm) 2D pooling.
      4. Create a channel-wise gate from pooled feature statistics via LogSigmoid and apply multiplicative gating.
      5. Restore original channel/depth split, aggregate across depth, and final L2 normalization.

    This model intentionally mixes nn.AvgPool3d, nn.LPPool2d and nn.LogSigmoid with tensor reshapes and norms
    to create a nontrivial computation graph while following PyTorch best practices.
    """
    def __init__(self,
                 lp_p: int = lp_norm_type,
                 avg_k: tuple = avgpool_kernel,
                 lp_k: int = lppool_kernel,
                 eps_val: float = eps):
        super(Model, self).__init__()
        # 3D average pooling to reduce spatial resolution
        self.avgpool3d = nn.AvgPool3d(kernel_size=avg_k, stride=avg_k)
        # LPPool2d with specified p-norm and kernel
        self.lppool2d = nn.LPPool2d(norm_type=lp_p, kernel_size=lp_k, stride=1)
        # LogSigmoid for gating nonlinearity
        self.logsigmoid = nn.LogSigmoid()
        self.eps = eps_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, D, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H_out, W_out) after pooling, gating and normalization.
        """
        # Step 1: Reduce spatial resolution with AvgPool3d
        out = self.avgpool3d(x)  # shape: (B, C, D2, H2, W2)
        B, C, D2, H2, W2 = out.shape

        # Step 2: Compute per-sample, per-channel L2 norm across spatial dims and normalize
        spatial_flat = out.view(B, C, -1)  # (B, C, D2*H2*W2)
        norms = torch.norm(spatial_flat, p=2, dim=2, keepdim=True)  # (B, C, 1)
        norms = norms.view(B, C, 1, 1, 1)  # (B, C, 1, 1, 1)
        out_normed = out / (norms + self.eps)  # normalized spatially

        # Step 3: Collapse depth into channel dimension to apply 2D LPPool
        # New channel dimension: C * D2
        out_2d = out_normed.reshape(B, C * D2, H2, W2)  # (B, C*D2, H2, W2)

        # Step 4: Apply LPPool2d
        pooled = self.lppool2d(out_2d)  # (B, C*D2, H3, W3)
        _, CD, H3, W3 = pooled.shape  # CD == C * D2

        # Step 5: Create a gate per (B, channel) by spatial averaging then applying LogSigmoid
        gate = pooled.mean(dim=[2, 3], keepdim=True)  # (B, C*D2, 1, 1)
        gate_act = self.logsigmoid(gate)  # (B, C*D2, 1, 1)

        # Step 6: Apply multiplicative gating
        gated = pooled * gate_act  # (B, C*D2, H3, W3)

        # Step 7: Restore (C, D2) split and aggregate across depth
        gated_5d = gated.view(B, C, D2, H3, W3)  # (B, C, D2, H3, W3)
        aggregated = gated_5d.mean(dim=2)  # aggregate across depth -> (B, C, H3, W3)

        # Step 8: Final L2 normalization per sample to stabilize scale
        final_flat = aggregated.view(B, -1)  # (B, C*H3*W3)
        final_norm = torch.norm(final_flat, p=2, dim=1, keepdim=True)  # (B, 1)
        final_norm = final_norm.view(B, 1, 1, 1)  # (B, 1, 1, 1)
        out_final = aggregated / (final_norm + self.eps)  # (B, C, H3, W3)

        return out_final


def get_inputs():
    """
    Returns a list with a single input tensor matching the configured shapes:
      (batch_size, channels, depth, height, width)
    """
    x = torch.randn(batch_size, channels, depth, height, width)
    return [x]


def get_init_inputs():
    """
    Returns the initialization parameters for the Model constructor:
      (lp_p, avg_k, lp_k, eps)
    """
    return [lp_norm_type, avgpool_kernel, lppool_kernel, eps]