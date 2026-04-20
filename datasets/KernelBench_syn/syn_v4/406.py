import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex example combining Hardshrink, AvgPool2d and LazyBatchNorm2d in a branching pattern.

    Computation pattern:
      1. Apply Hardshrink to suppress small activations (non-linear sparsification).
      2. Form a residual branch capturing the subtracted components: residual = x - hardshrink(x).
      3. Concatenate the processed and residual branches along the channel dimension.
      4. Apply 2D average pooling to reduce spatial resolution.
      5. Apply LazyBatchNorm2d (lazy initialization of channel count on first forward).
      6. Compute a lightweight channel descriptor (global spatial average) and use a tanh-based gating
         to reweight the normalized activations (channel-wise attention-like scaling).
    The model intentionally doubles channels via concatenation to exercise the lazy batch norm initialization.
    """
    def __init__(self, shrink_lambda: float = 0.7, pool_kernel: int = 3, pool_stride: int = 2, pool_padding: int = 1):
        super(Model, self).__init__()
        # Non-linear sparsifier (element-wise)
        self.shrink = nn.Hardshrink(lambd=shrink_lambda)
        # Average pooling reduces spatial dimensions
        self.pool = nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_stride, padding=pool_padding)
        # Lazy BatchNorm2d: will infer num_features (channels) on the first forward call
        self.bn = nn.LazyBatchNorm2d()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor with shape (batch, channels, height, width)

        Returns:
            torch.Tensor: Output tensor after shrink -> branch -> pool -> batchnorm -> gated scaling
        """
        # 1) Non-linear shrinkage: suppress small-magnitude elements
        shrunk = self.shrink(x)  # shape: (B, C, H, W)

        # 2) Residual branch: capture what was removed by hardshrink
        residual = x - shrunk  # shape: (B, C, H, W)

        # 3) Concatenate along channel dim to create interactions between kept and removed parts
        combined = torch.cat([shrunk, residual], dim=1)  # shape: (B, 2*C, H, W)

        # 4) Spatial downsampling
        pooled = self.pool(combined)  # shape: (B, 2*C, H', W')

        # 5) Batch normalization (lazy-initialized to 2*C channels on first forward)
        normalized = self.bn(pooled)  # shape: (B, 2*C, H', W')

        # 6) Lightweight channel descriptor (global spatial average) used to reweight channels
        #    Keep dims for broadcasting: shape becomes (B, 2*C, 1, 1)
        channel_desc = normalized.mean(dim=(2, 3), keepdim=True)

        # Gating mechanism: small tanh-based scaling to keep values bounded and differentiable
        gating = torch.tanh(channel_desc * 2.0 + 0.1)

        # Reweight normalized activations by the gating signal (channel-wise)
        out = normalized * gating  # shape: (B, 2*C, H', W')

        return out

# Configuration / default sizes for input tensors
BATCH = 8
CHANNELS = 3
HEIGHT = 224
WIDTH = 224

def get_inputs():
    """
    Returns:
        list: Single element list containing an input tensor of shape (BATCH, CHANNELS, HEIGHT, WIDTH)
    """
    x = torch.randn(BATCH, CHANNELS, HEIGHT, WIDTH)
    return [x]

def get_init_inputs():
    """
    No special initialization inputs required (LazyBatchNorm initializes on first forward).
    """
    return []