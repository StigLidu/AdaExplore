import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex 3D-to-3D feature gating module that demonstrates:
    - Lazy Instance Normalization over 3D volumes (LazyInstanceNorm3d)
    - Depth-collapse followed by 2D Adaptive Max Pooling (AdaptiveMaxPool2d)
    - Upsampling back to original spatial resolution and gated non-linearity (Tanh)
    The module learns a scalar gating magnitude and uses the pooled spatial template
    to modulate the normalized volumetric features.
    """
    def __init__(self, pool_output_h: int, pool_output_w: int, scale_init: float = 1.0, eps: float = 1e-5):
        """
        Args:
            pool_output_h (int): Target height for AdaptiveMaxPool2d.
            pool_output_w (int): Target width for AdaptiveMaxPool2d.
            scale_init (float, optional): Initial value for the learnable scalar gate. Defaults to 1.0.
            eps (float, optional): Epsilon for InstanceNorm. Defaults to 1e-5.
        """
        super(Model, self).__init__()
        # Lazy instance norm will infer num_features from the first forward input
        self.inst_norm = nn.LazyInstanceNorm3d(eps=eps, affine=True, track_running_stats=False)
        # Adaptive pooling to obtain a compact spatial template per-channel
        self.adapt_pool = nn.AdaptiveMaxPool2d((pool_output_h, pool_output_w))
        # Activation used for gating
        self.tanh = nn.Tanh()
        # Learnable scale parameter for gating strength
        self.scale = nn.Parameter(torch.tensor(float(scale_init), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation:
        1. Normalize the 3D input per-instance (LazyInstanceNorm3d).
        2. Collapse the depth dimension by averaging to obtain a 2D feature map per channel.
        3. Apply AdaptiveMaxPool2d to extract a compact spatial template.
        4. Upsample the template back to original spatial size and broadcast along depth.
        5. Compute a gated modulation via tanh(scale * template) and apply to normalized features,
           using a residual connection to preserve original signal.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of same shape as input, modulated by the learned template gate.
        """
        # 1) Instance-normalize the volumetric input
        x_norm = self.inst_norm(x)  # (N, C, D, H, W)

        # Record original spatial sizes
        N, C, D, H, W = x_norm.shape

        # 2) Collapse depth dimension -> (N, C, H, W)
        spatial = torch.mean(x_norm, dim=2)  # average over depth

        # 3) Adaptive max pool to compact template -> (N, C, Hp, Wp)
        pooled = self.adapt_pool(spatial)

        # 4) Upsample pooled template back to original spatial resolution (H, W)
        # Use bilinear interpolation over (H, W) for the 4D tensor
        upsampled = F.interpolate(pooled, size=(H, W), mode='bilinear', align_corners=False)

        # 5) Broadcast across depth: (N, C, 1, H, W) -> (N, C, D, H, W)
        template_5d = upsampled.unsqueeze(2).expand(-1, -1, D, -1, -1)

        # 6) Compute gated modulation and apply with residual connection
        gate = self.tanh(self.scale * template_5d)  # in (-1,1), shaped (N,C,D,H,W)
        out = x_norm * (1.0 + gate)  # modulate normalized features, preserve original via +1

        return out

# Configuration variables for input generation
batch_size = 8
channels = 16
depth = 10
height = 64
width = 48
pool_output_h = 8
pool_output_w = 6
scale_init = 0.75

def get_inputs():
    """
    Returns:
        List[torch.Tensor]: A single input tensor shaped (batch_size, channels, depth, height, width)
                            with random normal initialization.
    """
    x = torch.randn(batch_size, channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization arguments for Model.__init__:
        [pool_output_h, pool_output_w, scale_init]
    """
    return [pool_output_h, pool_output_w, scale_init]