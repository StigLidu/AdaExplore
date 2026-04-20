import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex normalization-gating module that demonstrates:
    - LazyInstanceNorm2d (lazy-initialized instance normalization)
    - A small gating MLP using SiLU activation to produce channel-wise gates
    - Channel-wise modulation, followed by CELU nonlinearity and a residual connection
    - Global average pooling and a final projection head

    The model accepts a 4D tensor (N, C, H, W) and returns a projected vector per sample.
    """
    def __init__(self, num_channels: int, gating_hidden: int, proj_out: int):
        """
        Initialize submodules and parameters.

        Args:
            num_channels (int): Number of channels in the input tensor (C).
            gating_hidden (int): Hidden dimension for the gating MLP.
            proj_out (int): Output dimension of the final projection head.
        """
        super(Model, self).__init__()
        # Lazy instance norm will be initialized when the first input is seen
        self.inst_norm = nn.LazyInstanceNorm2d()  # lazy: num_features inferred on first forward

        # Gating MLP: channel summary -> hidden -> channel gates
        self.gate_fc1 = nn.Linear(num_channels, gating_hidden)
        self.silu = nn.SiLU()  # activation for gating hidden layer
        self.gate_fc2 = nn.Linear(gating_hidden, num_channels)

        # Nonlinear activation applied to modulated feature maps
        self.celu = nn.CELU()

        # Final projection from channel-pooled features to desired output dimension
        self.proj = nn.Linear(num_channels, proj_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (N, proj_out).
        """
        # 1) Instance normalize the input (per-sample, per-channel)
        x_norm = self.inst_norm(x)  # (N, C, H, W)

        # 2) Compute channel-wise global descriptors via spatial global average pooling
        gap = x_norm.mean(dim=[2, 3])  # (N, C)

        # 3) Gating MLP: project -> SiLU -> project back to channels
        g = self.gate_fc1(gap)         # (N, gating_hidden)
        g = self.silu(g)               # (N, gating_hidden)
        g = self.gate_fc2(g)           # (N, C)

        # 4) Convert gating logits to [0, 1] scale and apply as channel-wise multiplicative gates
        gate = torch.sigmoid(g).unsqueeze(-1).unsqueeze(-1)  # (N, C, 1, 1)
        gated = x_norm * gate  # (N, C, H, W)

        # 5) Non-linear transform and residual connection
        out_feat = self.celu(gated)   # (N, C, H, W)
        out_res = x + out_feat        # (N, C, H, W)

        # 6) Global summary and final projection
        summary = out_res.mean(dim=[2, 3])  # (N, C)
        output = self.proj(summary)         # (N, proj_out)

        return output

# Configuration (module-level)
batch_size = 8
num_channels = 48
height = 32
width = 20
gating_hidden = 128
proj_out = 256

def get_inputs():
    """
    Create realistic input tensors for the model.

    Returns:
        list: [x] where x has shape (batch_size, num_channels, height, width)
    """
    x = torch.randn(batch_size, num_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Return initialization parameters for the Model constructor.

    Returns:
        list: [num_channels, gating_hidden, proj_out]
    """
    return [num_channels, gating_hidden, proj_out]