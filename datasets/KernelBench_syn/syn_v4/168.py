import torch
import torch.nn as nn

"""
Complex PyTorch module combining LazyInstanceNorm2d, Mish activation, and Softmax2d.
The model:
 - Normalizes the input feature map with LazyInstanceNorm2d (lazy channel inference).
 - Applies Mish non-linearity.
 - Computes a per-pixel attention distribution across channels with Softmax2d.
 - Builds a spatial context by weighting activated features with that attention and summing across channels.
 - Produces a spatial gating map from the context and re-scales the activated features.
 - Adds a residual connection and returns a global pooled embedding (per-channel) as output.

Module-level configuration variables define the shapes for input generation.
"""

# Configuration
BATCH_SIZE = 8
CHANNELS = 64
HEIGHT = 56
WIDTH = 56

class Model(nn.Module):
    """
    Model combining LazyInstanceNorm2d, Mish activation, and Softmax2d into a spatial
    attention and gating pattern that produces a per-channel global descriptor.

    Forward signature:
        x: Tensor of shape (N, C, H, W)

    Returns:
        Tensor of shape (N, C) -- per-channel global pooled descriptor after processing.
    """
    def __init__(self):
        super(Model, self).__init__()
        # Lazy instance normalization - num_features inferred on first forward
        self.inorm = nn.LazyInstanceNorm2d()
        # Non-linear activation
        self.mish = nn.Mish()
        # Softmax over channels at each spatial location
        self.softmax2d = nn.Softmax2d()
        # A small learnable scalar to modulate the residual path (initialized to 1.0)
        self.residual_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass implementing the described pattern.

        Steps:
        1. Normalize per-instance across channels.
        2. Apply Mish non-linearity.
        3. Compute per-pixel channel attention with Softmax2d.
        4. Form a spatial context by weighted sum across channels.
        5. Produce a gating map (tanh) from the context and re-scale activated features.
        6. Add a scaled residual connection from the original input and return global pooled descriptor.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (N, C) representing a global per-channel descriptor.
        """
        # 1) Normalize (lazy init discovers C here)
        x_norm = self.inorm(x)  # (N, C, H, W)

        # 2) Non-linearity
        x_act = self.mish(x_norm)  # (N, C, H, W)

        # 3) Per-pixel attention across channels
        att = self.softmax2d(x_act)  # (N, C, H, W) softmax over channel dim at each (h,w)

        # 4) Spatial context: weighted sum across channels -> shape (N, 1, H, W)
        context = (att * x_act).sum(dim=1, keepdim=True)  # (N, 1, H, W)

        # 5) Gating map derived from context (bounded between -1 and 1)
        gating = torch.tanh(context)  # (N, 1, H, W)

        # Broadcast gating to channels and apply to activated features
        gated = x_act * gating  # (N, C, H, W)

        # 6) Scaled residual connection and final aggregation:
        #    Add scaled original input (residual) to preserve low-level info.
        out = gated + self.residual_scale * x  # (N, C, H, W)

        # Global average pooling over spatial dimensions to produce per-channel descriptors
        out_pooled = out.mean(dim=(2, 3))  # (N, C)

        return out_pooled

def get_inputs():
    """
    Create a representative input tensor.

    Returns:
        List containing one tensor of shape (BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)
    """
    x = torch.randn(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)
    return [x]

def get_init_inputs():
    """
    No special init parameters are needed (LazyInstanceNorm2d infers channels).
    """
    return []