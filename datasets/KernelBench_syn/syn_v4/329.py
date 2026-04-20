import torch
import torch.nn as nn
from typing import Tuple

class Model(nn.Module):
    """
    Complex 3D feature processing module that demonstrates a sequence of nonlinearities,
    adaptive pooling, channel mixing, gating, and a small residual-style combination.

    Computation pipeline (high level):
      - Apply randomized leaky ReLU (RReLU) to the input.
      - Adaptive average pool to a fixed 3D grid.
      - 1x1x1 convolution to mix channels.
      - LeakyReLU activation.
      - Flatten spatial dimensions and apply a learned gating (sigmoid of an FC).
      - Reshape gated features back to 3D and add a scaled residual of the activated conv output.
      - Global average over spatial dims to produce final per-channel outputs.

    This model uses nn.RReLU, nn.AdaptiveAvgPool3d, and nn.LeakyReLU as core layers.
    """
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        pool_output: Tuple[int, int, int],
        negative_slope: float = 0.01,
        rrelu_lower: float = 0.125,
        rrelu_upper: float = 0.333,
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            mid_channels (int): Number of channels after 1x1x1 convolution.
            pool_output (tuple): Output size (D_out, H_out, W_out) for AdaptiveAvgPool3d.
            negative_slope (float): Negative slope for LeakyReLU.
            rrelu_lower (float): Lower bound for RReLU.
            rrelu_upper (float): Upper bound for RReLU.
        """
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.pool_output = tuple(pool_output)
        self.negative_slope = negative_slope

        # Randomized leaky ReLU applied first (stochastic during training)
        self.rrelu = nn.RReLU(lower=rrelu_lower, upper=rrelu_upper, inplace=False)

        # Adaptive pooling to reduce spatial dims to a small fixed grid
        self.adaptive_pool = nn.AdaptiveAvgPool3d(self.pool_output)

        # 1x1x1 Conv to mix channels without changing spatial dims
        self.conv1x1 = nn.Conv3d(in_channels, mid_channels, kernel_size=1, bias=True)

        # Deterministic leaky ReLU after channel mixing
        self.leaky = nn.LeakyReLU(negative_slope=self.negative_slope, inplace=False)

        # Fully connected gating layer: input size depends on mid_channels * prod(pool_output)
        prod_spatial = self.pool_output[0] * self.pool_output[1] * self.pool_output[2]
        fc_input_dim = mid_channels * prod_spatial
        self.gate_fc = nn.Linear(fc_input_dim, fc_input_dim, bias=True)

        # Small learnable scalar to weight the residual/skip before final pooling
        self.alpha = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

        # Initialize weights with reasonable defaults
        nn.init.kaiming_uniform_(self.conv1x1.weight, a=self.negative_slope)
        if self.conv1x1.bias is not None:
            nn.init.zeros_(self.conv1x1.bias)
        nn.init.xavier_uniform_(self.gate_fc.weight)
        if self.gate_fc.bias is not None:
            nn.init.zeros_(self.gate_fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_channels, D, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (batch, mid_channels) after spatial pooling.
        """
        # 1) Stochastic non-linearity to introduce noise-robustness during training
        x_r = self.rrelu(x)  # shape: (B, C_in, D, H, W)

        # 2) Adaptive 3D average pooling to reduce spatial resolution deterministically
        x_p = self.adaptive_pool(x_r)  # shape: (B, C_in, d_out, h_out, w_out)

        # 3) Channel mixing with 1x1x1 conv
        x_c = self.conv1x1(x_p)  # shape: (B, mid_channels, d_out, h_out, w_out)

        # 4) Deterministic non-linearity
        x_a = self.leaky(x_c)  # shape unchanged

        # 5) Flatten spatial dimensions for gating
        B = x_a.shape[0]
        flat = x_a.view(B, -1)  # shape: (B, mid_channels * prod_spatial)

        # 6) Learned gating (sigmoid) producing per-feature gates
        gates = torch.sigmoid(self.gate_fc(flat))  # shape: (B, mid*prod)

        # 7) Apply gating to flattened features
        gated_flat = flat * gates  # elementwise modulation

        # 8) Reshape back to 3D feature map
        gated_map = gated_flat.view_as(x_a)  # shape: (B, mid_channels, d_out, h_out, w_out)

        # 9) Residual-style combination: gated features + scaled activated conv features
        combined = gated_map + self.alpha * x_a  # broadcast scalar alpha

        # 10) Global average over spatial dims -> per-channel descriptors
        out = combined.mean(dim=(2, 3, 4))  # shape: (B, mid_channels)

        # Final non-linearity for stability
        out = self.leaky(out)

        return out


# Configuration / default parameters for test inputs
batch_size = 8
in_channels = 32
mid_channels = 64
depth = 8
height = 16
width = 16
pool_output = (2, 2, 2)
negative_slope = 0.02

def get_inputs():
    """
    Generates a random input tensor matching the configured dimensions.

    Returns:
        list: single-element list containing the input tensor.
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization inputs required to construct the Model instance.

    Order matches Model.__init__ signature:
      [in_channels, mid_channels, pool_output, negative_slope]

    Returns:
        list: initialization parameters
    """
    return [in_channels, mid_channels, pool_output, negative_slope]