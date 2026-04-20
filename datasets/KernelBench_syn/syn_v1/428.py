import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Patch-based feature extractor that:
      - Applies replication padding to preserve spatial context,
      - Extracts sliding patches using unfold,
      - Projects each patch with a learned linear layer,
      - Applies a Softsign nonlinearity followed by ReLU,
      - Aggregates patch features by spatial mean,
      - Applies an elementwise gating vector (also passed as input) after Softsign.

    This creates a mixed pipeline using nn.ReplicationPad2d, nn.Linear, nn.Softsign,
    and nn.ReLU in a non-trivial combination that operates on per-patch representations.
    """
    def __init__(self, in_channels: int, kernel_size: int, out_features: int, pad: int = None):
        """
        Args:
            in_channels (int): Number of input channels in the image tensor.
            kernel_size (int): Spatial size of patches to extract (square).
            out_features (int): Dimensionality of the projected patch features.
            pad (int, optional): Size of replication padding to apply on all sides.
                                 If None, uses kernel_size // 2 to preserve spatial dims.
        """
        super(Model, self).__init__()
        if pad is None:
            pad = kernel_size // 2
        self.pad = nn.ReplicationPad2d(pad)
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        # Linear projection applied to flattened patches: (C * K * K) -> out_features
        self.proj = nn.Linear(in_channels * kernel_size * kernel_size, out_features, bias=True)
        # Nonlinearities
        self.softsign = nn.Softsign()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch, in_channels, H, W).
            gate (torch.Tensor): Gating tensor of shape (batch, out_features) used to
                                 modulate the aggregated features.

        Returns:
            torch.Tensor: Output tensor of shape (batch, out_features).
        """
        # 1) Replication padding to give neighbors at borders
        x_p = self.pad(x)  # (batch, C, H + 2*pad, W + 2*pad)

        # 2) Extract sliding patches: unfold -> (batch, C*K*K, L) where L = number of patches
        patches = F.unfold(x_p, kernel_size=self.kernel_size)  # (B, C*K*K, L)

        # 3) Reformat to have patch dimension on axis 1: (batch, L, C*K*K)
        patches = patches.transpose(1, 2)

        # 4) Linear projection per patch: (batch, L, out_features)
        proj = self.proj(patches)

        # 5) Nonlinear composition: Softsign then ReLU (elementwise)
        proj = self.softsign(proj)
        proj = self.relu(proj)

        # 6) Aggregate spatial (patch) features by mean across L -> (batch, out_features)
        agg = proj.mean(dim=1)

        # 7) Gate the aggregated features: apply Softsign to gate and multiply
        gated = self.softsign(gate) * agg  # broadcasting elementwise for (batch, out_features)

        return gated

# Configuration / default sizes
batch_size = 8
in_channels = 3
height = 128
width = 128
kernel_size = 5
out_features = 256
pad = 2  # explicit pad; keeps spatial size same when kernel_size=5

def get_inputs():
    """
    Returns input tensors matching the expected forward signature:
      - x: random image tensor (batch, in_channels, H, W)
      - gate: random gating tensor (batch, out_features)
    """
    x = torch.randn(batch_size, in_channels, height, width)
    gate = torch.randn(batch_size, out_features)
    return [x, gate]

def get_init_inputs():
    """
    Returns constructor arguments for the Model:
      [in_channels, kernel_size, out_features, pad]
    """
    return [in_channels, kernel_size, out_features, pad]