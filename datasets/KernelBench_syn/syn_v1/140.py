import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    A composite model that:
    - Applies AlphaDropout for regularization,
    - Upsamples the spatial resolution using bilinear interpolation,
    - Collapses spatial dimensions into a 1D sequence per channel and applies AdaptiveMaxPool1d,
    - Projects the pooled per-channel features to a final output vector.

    This demonstrates combining dropout, upsampling, and pooling layers in a nontrivial data-flow.
    """
    def __init__(self,
                 in_channels: int,
                 upsample_scale: int,
                 pooled_length: int,
                 dropout_p: float,
                 out_features: int):
        """
        Initializes the composite model.

        Args:
            in_channels (int): Number of input channels.
            upsample_scale (int): Integer scale factor for bilinear upsampling.
            pooled_length (int): Output length for AdaptiveMaxPool1d.
            dropout_p (float): Probability for AlphaDropout.
            out_features (int): Size of the final output feature vector.
        """
        super(Model, self).__init__()
        # Regularization that preserves self-normalizing properties for SELU-like activations
        self.alpha_dropout = nn.AlphaDropout(p=dropout_p)

        # Bilinear upsampling; scale_factor expects int or tuple
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=upsample_scale)

        # Adaptive max pooling along the flattened spatial dimension (L)
        # We'll reshape (B, C, H', W') -> (B, C, L) and then pool to (B, C, pooled_length)
        self.adaptive_pool = nn.AdaptiveMaxPool1d(output_size=pooled_length)

        # A small projection head that maps per-channel pooled features to a compact vector
        self.proj = nn.Linear(in_channels * pooled_length, out_features)

        # A learnable scale parameter to modulate features after pooling
        self.register_parameter("post_scale", nn.Parameter(torch.ones(1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Steps:
            1. Apply AlphaDropout to the input tensor.
            2. Bilinearly upsample spatial resolution.
            3. Apply a non-linearity (ELU) to introduce non-linear mixing.
            4. Reshape to (B, C, L) where L = H'*W' and apply AdaptiveMaxPool1d.
            5. Elementwise scale and flatten channels+pooled_length.
            6. Project to final output vector.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, out_features).
        """
        # 1. Regularize
        x = self.alpha_dropout(x)

        # 2. Upsample spatially
        x = self.upsample(x)  # shape: (B, C, H', W')

        # 3. Non-linear activation for richer representation
        x = F.elu(x)

        # 4. Collapse spatial dims to a sequence for 1D pooling
        B, C, H_prime, W_prime = x.shape
        L = H_prime * W_prime
        x = x.reshape(B, C, L)  # (B, C, L)

        # 5. Adaptive max pool to fixed-length per channel
        x = self.adaptive_pool(x)  # (B, C, pooled_length)

        # 6. Modulate and flatten for projection
        x = x * self.post_scale  # broadcast scaling
        x = x.flatten(start_dim=1)  # (B, C * pooled_length)

        # 7. Final linear projection
        out = self.proj(x)  # (B, out_features)
        return out


# Module-level configuration
batch_size = 8
in_channels = 3
height = 16
width = 16
upsample_scale = 3
pooled_length = 8
dropout_p = 0.1
out_features = 128

def get_inputs():
    """
    Returns a list containing a single input tensor conforming to the configured shapes.
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters for Model in the same order as __init__.
    """
    return [in_channels, upsample_scale, pooled_length, dropout_p, out_features]