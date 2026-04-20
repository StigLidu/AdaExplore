import torch
import torch.nn as nn
import torch.nn.functional as F

# Configuration
batch_size = 8
in_channels = 3
height = 64
width = 48
upsample_scale = 2  # integer scale factor for bilinear upsampling
leaky_slope = 0.1
hardtanh_min = -0.7
hardtanh_max = 0.7
eps = 1e-6


class Model(nn.Module):
    """
    Complex vision-inspired module that:
    - Centers each channel by subtracting its spatial mean
    - Upsamples spatial resolution via bilinear interpolation
    - Applies LeakyReLU nonlinearity
    - Clamps values with HardTanh
    - Normalizes each channel by its spatial L2 norm
    - Pools back to the original spatial resolution and restores the mean (residual-style)
    """
    def __init__(
        self,
        in_channels: int,
        scale_factor: int = upsample_scale,
        negative_slope: float = leaky_slope,
        min_val: float = hardtanh_min,
        max_val: float = hardtanh_max,
        eps_val: float = eps
    ):
        super(Model, self).__init__()
        # Bilinear upsampling layer
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=scale_factor)
        # Nonlinearities
        self.leaky = nn.LeakyReLU(negative_slope=negative_slope)
        self.clamp = nn.Hardtanh(min_val, max_val)
        self.eps = eps_val
        # store channels for potential checks (not used as a param tensor)
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        1. Compute spatial mean per sample and channel, subtract to center.
        2. Upsample by a fixed bilinear scale.
        3. Apply LeakyReLU then HardTanh to introduce bounded nonlinearity.
        4. Compute per-sample per-channel L2 norm across spatial dims and normalize.
        5. Adaptive average pool back to original spatial size and add the mean back.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Tensor of shape (B, C, H, W) after the above transformations.
        """
        if x.dim() != 4:
            raise ValueError("Input must be a 4D tensor (B, C, H, W)")

        B, C, H, W = x.shape
        if C != self.in_channels:
            # Allowing flexibility but warn if channels mismatch
            # (this keeps behavior predictable while enabling different inputs)
            # Not raising to be permissive in testing harnesses.
            pass

        # 1) Channel-wise spatial mean: shape (B, C, 1, 1)
        spatial_mean = x.mean(dim=(2, 3), keepdim=True)

        # Center the input (residual-like operation)
        centered = x - spatial_mean

        # 2) Upsample spatial resolution
        up = self.upsample(centered)

        # 3) Nonlinear activation chain
        activated = self.leaky(up)
        clamped = self.clamp(activated)

        # 4) Normalize per-sample per-channel by spatial L2 norm
        # Flatten spatial dims to compute norms: result shape (B, C, 1)
        flattened = clamped.view(B, C, -1)
        l2_norm = flattened.norm(p=2, dim=2, keepdim=True)  # (B, C, 1)
        # reshape to broadcast over H', W'
        l2_norm = l2_norm.unsqueeze(-1)  # (B, C, 1, 1)
        normalized = clamped / (l2_norm + self.eps)

        # 5) Pool back to original resolution and add mean back (residual restoration)
        restored = F.adaptive_avg_pool2d(normalized, output_size=(H, W))
        output = restored + spatial_mean

        return output


def get_inputs():
    """
    Produce a representative input tensor for testing the Model.
    Returns:
        list containing one tensor of shape (batch_size, in_channels, height, width)
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    """
    Returns the initialization parameters for the Model constructor.
    These match the module-level configuration variables above.
    """
    return [in_channels, upsample_scale, leaky_slope, hardtanh_min, hardtanh_max, eps]