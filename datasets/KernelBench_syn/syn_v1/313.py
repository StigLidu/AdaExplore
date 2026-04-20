import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex vision-oriented module that upsamples an input feature map,
    performs channel-wise centering (subtract spatial mean), applies a clipped
    activation (ReLU6), uses AlphaDropout for regularization, reduces spatial
    dimensions via average pooling, and finally L2-normalizes the channel
    descriptors per-example.

    This module demonstrates a pipeline combining UpsamplingNearest2d,
    ReLU6, and AlphaDropout alongside standard tensor operations to produce
    robust per-channel descriptors.
    """
    def __init__(self, scale_factor: int, dropout_p: float, eps: float = 1e-6):
        """
        Args:
            scale_factor (int): Upsampling scale factor for height and width.
            dropout_p (float): Probability of an element to be zeroed in AlphaDropout.
            eps (float): Small epsilon for numerical stability in L2 normalization.
        """
        super(Model, self).__init__()
        # Vision upsampling layer (nearest neighbor)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=scale_factor)
        # Clipped nonlinear activation
        self.relu6 = nn.ReLU6()
        # AlphaDropout for self-normalizing networks behavior
        self.alpha_dropout = nn.AlphaDropout(p=dropout_p)
        # Small constant for numerical stability during normalization
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        1. Upsample spatial resolution by scale_factor using nearest neighbor.
        2. Compute the spatial mean (per sample, per channel) and center the feature map.
        3. Apply ReLU6 activation.
        4. Apply AlphaDropout for regularization.
        5. Average-pool across spatial dimensions to obtain per-channel descriptors.
        6. L2-normalize the descriptors per-sample.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: L2-normalized per-channel descriptors of shape (B, C).
        """
        # 1) Upsample
        x_up = self.upsample(x)

        # 2) Channel-wise centering using spatial mean (keep dims for broadcasting)
        spatial_mean = x_up.mean(dim=(2, 3), keepdim=True)
        x_centered = x_up - spatial_mean

        # 3) Clipped activation
        x_activated = self.relu6(x_centered)

        # 4) AlphaDropout (in-place semantics avoided)
        x_dropped = self.alpha_dropout(x_activated)

        # 5) Spatial average pooling -> per-channel descriptors (B, C)
        descriptors = x_dropped.mean(dim=(2, 3))

        # 6) L2-normalize per sample across channels
        l2_norm = torch.sqrt(torch.sum(descriptors * descriptors, dim=1, keepdim=True) + self.eps)
        normalized = descriptors / l2_norm

        return normalized

# Module-level configuration variables
batch_size = 8
channels = 64
height = 16
width = 12
scale_factor = 2          # Upsampling factor for both H and W
dropout_p = 0.1           # AlphaDropout probability
eps = 1e-6                # Epsilon for numerical stability

def get_inputs():
    """
    Returns the primary input tensor for the model:
    - A random float tensor shaped (batch_size, channels, height, width)
    """
    x = torch.randn(batch_size, channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for the Model constructor in the correct order:
    [scale_factor, dropout_p, eps]
    """
    return [scale_factor, dropout_p, eps]