import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex 3D processing module that:
    - Applies a channel-wise adaptive scaling based on per-channel global averages.
    - Performs Lp pooling over 3D spatial dimensions to reduce resolution.
    - Applies randomized leaky ReLU (RReLU) activation.
    - Applies 3D channel dropout (Dropout3d).
    - Projects channels from in_channels -> out_channels via a learnable linear map applied per-spatial-location.

    Forward input:
        x: Tensor of shape (B, C, D, H, W)

    Output:
        Tensor of shape (B, out_channels, D', H', W') where D', H', W' are reduced by the pooling layer.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        lp_norm: int = 2,
        pool_kernel: tuple = (3, 3, 3),
        pool_stride: tuple = (2, 2, 2),
        dropout_p: float = 0.2,
        rrelu_lower: float = 0.125,
        rrelu_upper: float = 0.333,
        alpha: float = 0.5,
    ):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Adaptive scaling coefficient (fixed scalar)
        self.alpha = float(alpha)

        # Lp pooling reduces spatial resolution in D,H,W
        # nn.LPPool3d expects (norm_type, kernel_size, stride=None, ceil_mode=False)
        self.pool = nn.LPPool3d(lp_norm, kernel_size=pool_kernel, stride=pool_stride)

        # Randomized leaky ReLU
        self.rrelu = nn.RReLU(lower=rrelu_lower, upper=rrelu_upper, inplace=False)

        # Drop entire channels (3D dropout)
        self.dropout = nn.Dropout3d(p=dropout_p)

        # Learnable channel projection weight and bias: projecting C -> out_channels
        # We represent it as a weight matrix of shape (out_channels, in_channels)
        self.proj_weight = nn.Parameter(torch.randn(out_channels, in_channels) * (1.0 / (in_channels ** 0.5)))
        self.proj_bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation:
         1. Compute per-sample per-channel global average over D,H,W and do channel-wise adaptive scaling:
                x_scaled = x * (1 + alpha * channel_avg)
         2. Apply Lp pooling on 3D spatial dims.
         3. Apply RReLU activation.
         4. Apply Dropout3d.
         5. Project channels via einsum using proj_weight to obtain out_channels.
        """
        # x: (B, C, D, H, W)
        # 1) Channel-wise global average: shape (B, C, 1, 1, 1)
        channel_avg = x.mean(dim=(2, 3, 4), keepdim=True)

        # Adaptive scaling
        x = x * (1.0 + self.alpha * channel_avg)

        # 2) Lp pooling reduces spatial dims
        x = self.pool(x)  # shape becomes (B, C, D', H', W')

        # 3) Randomized leaky ReLU
        x = self.rrelu(x)

        # 4) Dropout3d (drops entire channels randomly)
        x = self.dropout(x)

        # 5) Channel projection: apply linear map to channels for each spatial location
        # Use einsum to apply weight across the channel dimension:
        # x: (B, C, D', H', W'), proj_weight: (O, C) -> result (B, O, D', H', W')
        x = torch.einsum('b c d h w, o c -> b o d h w', x, self.proj_weight)
        x = x + self.proj_bias.view(1, -1, 1, 1, 1)

        return x

# Configuration variables
BATCH = 4
IN_CHANNELS = 8
OUT_CHANNELS = 16
DEPTH = 16
HEIGHT = 32
WIDTH = 32

# Pooling parameters
LP_NORM = 2
POOL_KERNEL = (3, 3, 3)
POOL_STRIDE = (2, 2, 2)

# Activation / dropout params
DROPOUT_P = 0.25
RRELU_LOWER = 0.125
RRELU_UPPER = 0.333

# Adaptive scaling coefficient
ALPHA = 0.7

def get_inputs():
    """
    Returns example inputs for the forward pass.
    """
    x = torch.randn(BATCH, IN_CHANNELS, DEPTH, HEIGHT, WIDTH)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for constructing the Model.
    Order matches Model.__init__ signature:
    (in_channels, out_channels, lp_norm, pool_kernel, pool_stride, dropout_p, rrelu_lower, rrelu_upper, alpha)
    """
    return [IN_CHANNELS, OUT_CHANNELS, LP_NORM, POOL_KERNEL, POOL_STRIDE, DROPOUT_P, RRELU_LOWER, RRELU_UPPER, ALPHA]