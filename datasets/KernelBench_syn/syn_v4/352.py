import torch
import torch.nn as nn

# Module-level configuration
batch_size = 16
in_channels = 3
mid_channels = 256
H = 32
W = 32
up_scale = 2
out_features = 4096
hardshrink_lambda = 0.75  # threshold for Hardshrink


class Model(nn.Module):
    """
    A moderately complex vision-to-vector module illustrating a mix of convolution,
    nearest-neighbor upsampling, sparsifying activation (Hardshrink), spatial pooling,
    and a dense projection with a final LogSigmoid.

    Computation pipeline:
      1. 2D convolution for local feature extraction
      2. ReLU nonlinearity
      3. Nearest-neighbor upsampling (spatial enlargement)
      4. Hardshrink to sparsify small activations
      5. Adaptive average pooling to a compact spatial representation
      6. Linear projection to a high-dimensional vector
      7. LogSigmoid applied elementwise to the output vector

    This is functionally distinct from simple elementwise activations or pure matmuls.
    """
    def __init__(
        self,
        in_ch: int = in_channels,
        mid_ch: int = mid_channels,
        out_dim: int = out_features,
        scale_factor: int = up_scale,
        hardshrink_lambd: float = hardshrink_lambda,
    ):
        super().__init__()
        # Local feature extractor
        self.conv = nn.Conv2d(in_ch, mid_ch, kernel_size=3, stride=1, padding=1, bias=True)
        # Upsampling layer (nearest neighbor)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=scale_factor)
        # Sparsifying activation
        self.hardshrink = nn.Hardshrink(lambd=hardshrink_lambd)
        # Reduce spatial dims to a single vector per channel
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # Final projection to a high-dimensional vector
        self.fc = nn.Linear(mid_ch, out_dim)
        # Final stabilized activation
        self.logsigmoid = nn.LogSigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (B, out_features) with LogSigmoid applied.
        """
        # Feature extraction
        x = self.conv(x)                # (B, mid_ch, H, W)
        x = torch.relu(x)               # nonlinearity

        # Upsample spatial resolution
        x = self.upsample(x)            # (B, mid_ch, H*scale, W*scale)

        # Sparsify small activations (encourages zeros)
        x = self.hardshrink(x)          # elementwise

        # Global-ish pooling to vector
        x = self.pool(x)                # (B, mid_ch, 1, 1)
        x = x.view(x.size(0), -1)       # (B, mid_ch)

        # Project to high-dimensional feature vector
        x = self.fc(x)                  # (B, out_features)

        # Stabilize/log-scale outputs
        x = self.logsigmoid(x)          # (B, out_features)
        return x


def get_inputs():
    """
    Generates a batch of random images according to module-level configuration.

    Returns:
        list: single-element list containing the input tensor [x]
    """
    x = torch.randn(batch_size, in_channels, H, W)
    return [x]


def get_init_inputs():
    """
    Returns initialization configuration used to construct the Model.
    This can be used by test harnesses to initialize the Model with the same parameters.

    Returns:
        list: [in_channels, mid_channels, out_features, up_scale, hardshrink_lambda]
    """
    return [in_channels, mid_channels, out_features, up_scale, hardshrink_lambda]