import torch
import torch.nn as nn

# Configuration
batch_size = 8
channels = 64
height = 56
width = 56

pool_size = 7          # AdaptiveMaxPool2d output size (pool_size x pool_size)
proj_dim = 1024        # Output dimensionality after projection
negative_slope = 0.02  # LeakyReLU negative slope
shrink_lambda = 0.2    # Hardshrink lambda threshold

class Model(nn.Module):
    """
    Complex module that combines adaptive max pooling, LeakyReLU, Hardshrink,
    channel gating, and a final linear projection.

    Forward signature:
        (x: Tensor, gating: Tensor) -> Tensor
    where:
        x: Tensor of shape (B, C, H, W)
        gating: Tensor of shape (B, C) or (B, C, 1, 1)
    """
    def __init__(
        self,
        in_channels: int,
        pool_out: int,
        out_dim: int,
        neg_slope: float = 0.01,
        hardshrink_lambda: float = 0.5,
    ):
        super(Model, self).__init__()
        # Spatial reduction to a fixed grid
        self.pool = nn.AdaptiveMaxPool2d((pool_out, pool_out))
        # Non-linearities
        self.leaky = nn.LeakyReLU(negative_slope=neg_slope, inplace=False)
        self.hardshrink = nn.Hardshrink(lambd=hardshrink_lambda)
        # Projection parameters: project flattened pooled features to out_dim
        flattened_dim = in_channels * pool_out * pool_out
        self.proj_weight = nn.Parameter(torch.randn(flattened_dim, out_dim) * (1.0 / (flattened_dim ** 0.5)))
        self.proj_bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor, gating: torch.Tensor) -> torch.Tensor:
        """
        Pipeline:
          1. AdaptiveMaxPool2d to (P, P)
          2. LeakyReLU activation
          3. Hardshrink (sparsifies small activations)
          4. Channel-wise gating (sigmoid of provided gating tensor)
          5. Flatten and linear projection (matrix multiply + bias)
          6. Final tanh nonlinearity for bounded outputs

        Args:
            x: (B, C, H, W)
            gating: (B, C) or (B, C, 1, 1)

        Returns:
            Tensor of shape (B, out_dim)
        """
        # Step 1: spatial adaptive max pooling -> (B, C, P, P)
        pooled = self.pool(x)

        # Step 2: non-linearity
        activated = self.leaky(pooled)

        # Step 3: sparsify small values
        shrunk = self.hardshrink(activated)

        # Step 4: apply channel-wise gating
        # Normalize gating shape to (B, C, 1, 1)
        if gating.dim() == 2:
            gate = gating.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        elif gating.dim() == 3:
            gate = gating.unsqueeze(-1)  # (B, C, 1, 1)
        else:
            gate = gating  # assume already (B, C, 1, 1)

        gate_sig = torch.sigmoid(gate)
        gated = shrunk * gate_sig  # broadcast over spatial dims

        # Step 5: flatten and project
        B = gated.shape[0]
        flat = gated.view(B, -1)  # (B, C * P * P)
        out = flat @ self.proj_weight + self.proj_bias  # (B, out_dim)

        # Step 6: bounded non-linearity
        return torch.tanh(out)

def get_inputs():
    """
    Returns:
      - x: random input tensor of shape (batch_size, channels, height, width)
      - gating: random gating tensor of shape (batch_size, channels)
    """
    x = torch.randn(batch_size, channels, height, width)
    gating = torch.randn(batch_size, channels)
    return [x, gating]

def get_init_inputs():
    """
    Returns the initialization arguments for Model(in_channels, pool_out, out_dim, neg_slope, hardshrink_lambda)
    """
    return [channels, pool_size, proj_dim, negative_slope, shrink_lambda]