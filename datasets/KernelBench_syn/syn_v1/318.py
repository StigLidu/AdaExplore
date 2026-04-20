import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex feature-wise adaptive normalization and gating module.

    This module:
    - Applies Instance Normalization to the input.
    - Computes a global per-channel summary (mean over spatial dims).
    - Uses learnable per-channel parameters to produce an adaptive scale via Softplus.
    - Converts that scale into a gating factor with Hardsigmoid.
    - Mixes normalized and raw input using the gate (residual-style).
    - Applies a final per-channel affine transform and a Softplus+Hardsigmoid nonlinearity.

    The constructor accepts parameters required to configure InstanceNorm2d and whether
    an internal bias should be used for the initial scaling transform.
    """
    def __init__(self, num_channels: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True, use_bias_in_scale: bool = True):
        """
        Args:
            num_channels (int): Number of channels for input tensors.
            eps (float): A value added to the denominator for numerical stability in InstanceNorm2d.
            momentum (float): The value used for the running_mean and running_var computation in InstanceNorm2d.
            affine (bool): If True, InstanceNorm2d has learnable affine parameters.
            use_bias_in_scale (bool): If True, include a per-channel bias in the adaptive scale computation.
        """
        super(Model, self).__init__()
        self.num_channels = num_channels
        # Primary normalization
        self.inst_norm = nn.InstanceNorm2d(num_channels, eps=eps, momentum=momentum, affine=affine)
        # Non-linearities
        self.softplus = nn.Softplus()
        self.hardsigmoid = nn.Hardsigmoid()
        # Learnable parameters for creating adaptive channel-wise scale from global summary
        # These are intentionally small initialized to avoid exploding activations
        self.channel_scale = nn.Parameter(torch.randn(num_channels) * 0.05)  # multiplies global mean
        if use_bias_in_scale:
            self.channel_bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter('channel_bias', None)
        # Final per-channel affine transform after mixing normalized/raw features
        self.res_scale = nn.Parameter(torch.ones(num_channels))
        self.res_shift = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of same shape (B, C, H, W).
        """
        B, C, H, W = x.shape
        assert C == self.num_channels, f"Expected input with {self.num_channels} channels but got {C}"

        # 1) Instance normalization (per-instance, per-channel)
        x_norm = self.inst_norm(x)  # (B, C, H, W)

        # 2) Global summary (per-sample, per-channel): mean across spatial dimensions
        global_mean = x_norm.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)

        # 3) Compute adaptive scale from global summary and learnable per-channel parameters
        #    channel_scale: (C,) -> reshape to (1,C,1,1) to broadcast against (B,C,1,1)
        ch_scale = self.channel_scale.view(1, C, 1, 1)
        bias = self.channel_bias.view(1, C, 1, 1) if self.channel_bias is not None else 0.0
        adaptive_raw = global_mean * ch_scale + bias  # (B, C, 1, 1)

        # 4) Smooth positive scaling using Softplus, then compress to gating using Hardsigmoid
        adaptive_scale = self.softplus(adaptive_raw)  # positive scale
        gate = self.hardsigmoid(adaptive_scale)  # in [0,1], (B, C, 1, 1)

        # 5) Residual-style mixing: favor normalized features when gate ~1, raw input when gate ~0
        mixed = x_norm * gate + x * (1.0 - gate)  # (B, C, H, W)

        # 6) Final per-channel affine transform followed by a gentle nonlinearity and compression
        res_s = self.res_scale.view(1, C, 1, 1)
        res_b = self.res_shift.view(1, C, 1, 1)
        out = mixed * res_s + res_b
        out = self.softplus(out)
        out = self.hardsigmoid(out)

        return out

# Configuration / default sizes
batch_size = 8
channels = 64
height = 32
width = 32
eps = 1e-5
momentum = 0.1
affine = True
use_bias_in_scale = True

def get_inputs():
    """
    Returns:
        list: Single-element list containing an input tensor of shape (batch_size, channels, height, width).
    """
    x = torch.randn(batch_size, channels, height, width, dtype=torch.float32)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for the Model constructor in the following order:
        num_channels, eps, momentum, affine, use_bias_in_scale
    """
    return [channels, eps, momentum, affine, use_bias_in_scale]