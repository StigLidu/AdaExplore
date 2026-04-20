import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex model that combines Lp-pooling, Softsign activation, adaptive max pooling,
    and a small channel-attention mechanism implemented with Linear layers.

    Computation pipeline:
        1. Apply LPPool2d to perform a learned power-average spatial downsampling.
        2. Apply Softsign non-linearity.
        3. Apply AdaptiveMaxPool2d to collapse spatial dimensions to a fixed output size.
        4. Compute channel descriptors by spatial averaging and pass through a bottleneck
           MLP (Linear -> Softsign -> Linear -> Sigmoid) to obtain per-channel gates.
        5. Modulate the pooled tensor with gates and combine with a small learned residual scale.

    This creates a spatial + channel modulation pattern distinct from simple reductions or
    single-layer transformations.
    """
    def __init__(
        self,
        channels: int,
        p: float = 2.0,
        lp_kernel: int = 3,
        lp_stride: int = 2,
        adaptive_out: tuple = (2, 2),
    ):
        """
        Args:
            channels (int): Number of channels in the input tensor.
            p (float): The p-norm for LPPool2d (norm_type).
            lp_kernel (int): Kernel size for LPPool2d.
            lp_stride (int): Stride for LPPool2d.
            adaptive_out (tuple): Output spatial size (H_out, W_out) for AdaptiveMaxPool2d.
        """
        super(Model, self).__init__()
        self.channels = channels

        # Spatial pooling: Lp pooling followed by adaptive max pooling
        # LPPool2d expects an integer norm_type; round/convert p to int for norm_type
        self.lp_pool = nn.LPPool2d(int(max(1, round(p))), kernel_size=lp_kernel, stride=lp_stride)
        self.softsign = nn.Softsign()
        self.adaptive_pool = nn.AdaptiveMaxPool2d(adaptive_out)

        # Channel attention: small bottleneck MLP
        hidden = max(1, channels // 4)
        self.att_fc1 = nn.Linear(channels, hidden, bias=True)
        self.att_fc2 = nn.Linear(hidden, channels, bias=True)

        # Learnable residual per-channel scale
        self.channel_scale = nn.Parameter(torch.ones(channels), requires_grad=True)

        # Initialize weights for stability
        nn.init.kaiming_uniform_(self.att_fc1.weight, a=0.1)
        nn.init.zeros_(self.att_fc1.bias)
        nn.init.kaiming_uniform_(self.att_fc2.weight, a=0.1)
        nn.init.zeros_(self.att_fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H_out, W_out)
        """
        # Step 1: Lp pooling reduces spatial resolution with a power-average
        x_lp = self.lp_pool(x)  # (B, C, H_lp, W_lp)

        # Step 2: Softsign activation to introduce non-linearity (bounded)
        x_act = self.softsign(x_lp)

        # Step 3: Adaptive max pool to fixed spatial output
        x_adapt = self.adaptive_pool(x_act)  # (B, C, H_out, W_out)

        # Step 4: Channel gating
        # Compute channel descriptors by spatial average
        B, C, H_out, W_out = x_adapt.shape
        channel_desc = x_adapt.mean(dim=(2, 3))  # (B, C)

        # Bottleneck MLP with Softsign non-linearity
        att = self.att_fc1(channel_desc)         # (B, hidden)
        att = self.softsign(att)
        att = self.att_fc2(att)                  # (B, C)
        gate = torch.sigmoid(att).view(B, C, 1, 1)  # (B, C, 1, 1)

        # Step 5: Modulate and add learned residual scaling
        scale = self.channel_scale.view(1, C, 1, 1)
        out = x_adapt * gate + x_adapt * scale

        return out

# Configuration / default sizes
batch_size = 8
channels = 64
height = 64
width = 64

# Lp pooling and adaptive pool configuration
p_norm = 2.0
lp_kernel = 3
lp_stride = 2
adaptive_output = (4, 4)

def get_inputs():
    """
    Returns example input tensors for the model forward.
    """
    x = torch.randn(batch_size, channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for the Model constructor:
        [channels, p, lp_kernel, lp_stride, adaptive_out]
    """
    return [channels, p_norm, lp_kernel, lp_stride, adaptive_output]