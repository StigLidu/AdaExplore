import torch
import torch.nn as nn

# Configuration / shape constants
BATCH = 8
CHANNELS = 64
HEIGHT = 32
WIDTH = 32

class Model(nn.Module):
    """
    Channel-mixing spatial transformer:
    - Applies Softmax2d to normalize channel responses at each spatial location.
    - Mixes channel information via a learnable channel-mixing matrix.
    - Applies randomized Leaky ReLU (RReLU) as a stochastic activation.
    - Adds a scaled residual connection from the original input and final ReLU.
    
    Forward pattern:
        x -> Softmax2d -> channel-mix (matmul) -> +bias -> RReLU -> +scaled skip -> ReLU -> out
    Input:
        x: Tensor of shape (BATCH, CHANNELS, HEIGHT, WIDTH)
    Initialization:
        mix_init: Tensor of shape (CHANNELS, CHANNELS) to initialize the mixing matrix
        bias_init: Tensor of shape (CHANNELS,) to initialize the per-channel bias
    """
    def __init__(self, mix_init: torch.Tensor, bias_init: torch.Tensor = None):
        super(Model, self).__init__()
        # Validate init shapes
        assert mix_init.ndim == 2 and mix_init.shape[0] == mix_init.shape[1], \
            "mix_init must be square matrix of shape (C, C)"
        C = mix_init.shape[0]
        self.C = C

        # Learnable channel mixing matrix and optional bias
        self.mix = nn.Parameter(mix_init.clone().float())
        if bias_init is not None:
            assert bias_init.ndim == 1 and bias_init.shape[0] == C, "bias_init must be shape (C,)"
            self.bias = nn.Parameter(bias_init.clone().float())
        else:
            # initialize bias to zero if not provided
            self.bias = nn.Parameter(torch.zeros(C, dtype=torch.float32))

        # Non-linearities
        self.soft2d = nn.Softmax2d()
        # Use randomized leaky ReLU with small bounds for stochastic behavior
        self.rrelu = nn.RReLU(lower=0.1, upper=0.3, inplace=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C, H, W)
        Returns:
            Tensor of same shape with channel-mixing and activations applied.
        """
        # Ensure expected channel dimension
        B, C, H, W = x.shape
        assert C == self.C, f"Input channels ({C}) must match mixing matrix ({self.C})"

        # 1) Normalize channel activations per spatial location
        s = self.soft2d(x)  # (B, C, H, W)

        # 2) Prepare for channel-wise linear mixing:
        #    reshape to (B, S, C) where S = H*W to perform batch matmul with mix matrix
        S = H * W
        s_flat = s.view(B, C, S).permute(0, 2, 1)  # (B, S, C)

        # 3) Channel mixing: for each (batch, spatial) vector of length C, multiply by mix^T
        #    result shape: (B, S, C)
        mixed_spatial = torch.matmul(s_flat, self.mix.t())

        # 4) Restore spatial layout and add per-channel bias
        mixed = mixed_spatial.permute(0, 2, 1).view(B, C, H, W)  # (B, C, H, W)
        mixed = mixed + self.bias.view(1, C, 1, 1)

        # 5) Apply randomized leaky ReLU
        activated = self.rrelu(mixed)

        # 6) Residual connection (scaled) and final ReLU
        out = self.relu(activated + 0.1 * x)

        return out

def get_inputs():
    """
    Returns sample input tensors for the forward pass.
    """
    x = torch.randn(BATCH, CHANNELS, HEIGHT, WIDTH)
    return [x]

def get_init_inputs():
    """
    Returns initialization tensors used to construct the Model.
    Here we provide a random mixing matrix and a small random bias vector.
    """
    # Initialize mixing matrix with small random values
    mix_init = torch.randn(CHANNELS, CHANNELS) * (1.0 / CHANNELS**0.5)
    bias_init = torch.randn(CHANNELS) * 0.01
    return [mix_init, bias_init]