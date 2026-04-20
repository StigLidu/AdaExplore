import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex module that:
      - Applies circular padding to a 4D input (N, C, H, W)
      - Extracts local patches with unfold (emulating a small convolution receptive field)
      - Projects patches into a hidden representation, applies SiLU non-linearity
      - Applies lazy BatchNorm1d across the hidden channels (in a (N, C_hidden, L) layout)
      - Aggregates statistics to produce per-channel gating scales which modulate the original input
    This produces a spatially-consistent channel gating mechanism informed by local patch features.
    """
    def __init__(self, in_channels: int, kernel_size: int = 3, hidden_dim: int = 64, pad: int = 1):
        """
        Args:
            in_channels (int): Number of input channels.
            kernel_size (int): Patch kernel size (assumed square). Default 3.
            hidden_dim (int): Dimensionality of the hidden projected patch features.
            pad (int): Circular padding applied to H and W dimensions. Default 1 (common for 3x3).
        """
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.hidden_dim = hidden_dim
        self.pad = pad

        # Circular padding layer (wrap-around along spatial dimensions)
        # padding order for CircularPad2d: (left, right, top, bottom) or single int
        self.pad_layer = nn.CircularPad2d((pad, pad, pad, pad))

        # Learnable linear projection from flattened patch -> hidden_dim
        # We store as Parameter to use F.linear (weight shape: out_features x in_features)
        in_feat = in_channels * (kernel_size * kernel_size)
        self.proj_weight = nn.Parameter(torch.randn(hidden_dim, in_feat) * (1.0 / (in_feat ** 0.5)))
        self.proj_bias = nn.Parameter(torch.zeros(hidden_dim))

        # SiLU activation (a smooth non-linearity)
        self.act = nn.SiLU()

        # Lazy BatchNorm1d: will infer num_features on first forward call from input's channel dim
        # We will pass inputs shaped (N, hidden_dim, L) so it will set num_features = hidden_dim
        self.bn = nn.LazyBatchNorm1d()

        # A small linear layer that maps aggregated hidden features -> per-input-channel scales
        self.scale_proj = nn.Linear(hidden_dim, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W)
        Returns:
            torch.Tensor: Modulated output tensor of shape (N, C, H, W)
        """
        # 1) Circular pad the input so patches at borders wrap around
        x_padded = self.pad_layer(x)  # (N, C, H + 2*pad, W + 2*pad)

        # 2) Extract local patches using unfold. Each patch flattened has size C * K * K.
        #    Output shape: (N, C*K*K, L) where L is number of sliding locations (H_out * W_out).
        patches = F.unfold(x_padded, kernel_size=self.kernel_size, stride=1, padding=0)  # (N, in_feat, L)
        # Permute to (N, L, in_feat) to apply linear mapping on last dim
        patches_t = patches.permute(0, 2, 1)  # (N, L, in_feat)

        # 3) Project each patch into a hidden representation and apply SiLU
        #    Use F.linear which expects weight shape (out_features, in_features)
        projected = F.linear(patches_t, self.proj_weight, self.proj_bias)  # (N, L, hidden_dim)
        activated = self.act(projected)  # (N, L, hidden_dim)

        # 4) Prepare for BatchNorm1d: (N, hidden_dim, L)
        activated_t = activated.permute(0, 2, 1)  # (N, hidden_dim, L)
        normalized = self.bn(activated_t)  # (N, hidden_dim, L)

        # 5) Aggregate per-hidden-channel statistics across spatial locations -> (N, hidden_dim)
        pooled = normalized.mean(dim=2)  # (N, hidden_dim)

        # 6) Compute per-input-channel gating scales and squash to (0,1)
        channel_logits = self.scale_proj(pooled)  # (N, in_channels)
        channel_scale = torch.sigmoid(channel_logits).unsqueeze(-1).unsqueeze(-1)  # (N, in_channels, 1, 1)

        # 7) Apply gating to the original (unpadded) input
        modulated = x * channel_scale  # (N, in_channels, H, W)

        # 8) Final non-linearity and a small residual connection to stabilize gradients
        out = self.act(modulated + 0.1 * x)

        return out

# Module-level configuration variables
batch_size = 8
channels = 32
height = 64
width = 64
kernel_size = 3
hidden_dim = 64
pad = 1

def get_inputs():
    """
    Returns a list containing a single input tensor matching the module-level configuration.
    """
    x = torch.randn(batch_size, channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for the Model: [in_channels, kernel_size, hidden_dim, pad]
    """
    return [channels, kernel_size, hidden_dim, pad]