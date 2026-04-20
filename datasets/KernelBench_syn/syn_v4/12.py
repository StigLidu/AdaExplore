import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A composite vision module that demonstrates a multi-step transformation using:
    - PixelUnshuffle to trade spatial resolution for channel capacity
    - Per-position linear projection to a hidden feature dimension
    - RMSNorm for normalization across the feature dimension
    - Hardsigmoid as a lightweight gating non-linearity
    - Global context pooling to produce a per-channel gate
    - Reconstruction back to the original image shape via PixelShuffle

    The model preserves input shape (batch, in_channels, H, W) and is functionally
    distinct from simple activations or single-layer blocks by combining spatial
    reshaping, per-position projections, normalization, and context gating.
    """
    def __init__(self, in_channels: int, downscale_factor: int, hidden_dim: int, eps: float = 1e-6):
        """
        Args:
            in_channels (int): Number of input image channels.
            downscale_factor (int): PixelUnshuffle downscale factor (r). Must divide height/width.
            hidden_dim (int): Hidden per-position feature dimension after the first linear projection.
            eps (float): Small epsilon value for RMSNorm stability.
        """
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.r = downscale_factor
        # PixelUnshuffle reduces spatial dims H,W by factor r and multiplies channels by r^2.
        self.unshuffle = nn.PixelUnshuffle(self.r)
        # After unshuffle, channel count becomes in_channels * (r^2)
        self.c_prime = in_channels * (self.r ** 2)

        # Per-position linear projection maps c_prime -> hidden_dim
        self.proj_in = nn.Linear(self.c_prime, hidden_dim, bias=True)

        # RMSNorm normalizes over the last dimension (hidden_dim)
        self.rmsnorm = nn.RMSNorm(hidden_dim, eps=eps)

        # Lightweight elementwise non-linearity
        self.act = nn.Hardsigmoid()

        # Context projection to compute per-channel gates from global pooled context
        self.context_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        # Output projection to reconstruct c_prime channels per spatial position
        self.proj_out = nn.Linear(hidden_dim, self.c_prime, bias=True)

        # PixelShuffle will upsample spatial dims by factor r and reduce channels by r^2
        # This will recover the original in_channels when applied to a tensor with c_prime channels.
        self.shuffle = nn.PixelShuffle(self.r)

        # Initialize weights with a stable scheme
        self._reset_parameters()

    def _reset_parameters(self):
        # Use Xavier uniform for linear layers and default for RMSNorm
        nn.init.xavier_uniform_(self.proj_in.weight)
        if self.proj_in.bias is not None:
            nn.init.zeros_(self.proj_in.bias)
        nn.init.xavier_uniform_(self.proj_out.weight)
        if self.proj_out.bias is not None:
            nn.init.zeros_(self.proj_out.bias)
        nn.init.xavier_uniform_(self.context_proj.weight)
        if self.context_proj.bias is not None:
            nn.init.zeros_(self.context_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
         1. PixelUnshuffle to reduce H,W and increase channels.
         2. Move channel to last dimension for per-position linear operations.
         3. Project per-position channels -> hidden_dim.
         4. Apply RMSNorm and Hardsigmoid activation.
         5. Produce a global context by average pooling over spatial dims and transform it
            to a gating vector via a small linear block and Hardsigmoid.
         6. Modulate per-position features by the broadcasted gate.
         7. Project back to the unshuffled channel dimension and PixelShuffle to restore original shape.

        Args:
            x (torch.Tensor): Input tensor of shape (B, in_channels, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, in_channels, H, W), same spatial dims as input.
        """
        # Step 1: reduce spatial resolution
        # x_un: (B, c_prime, H//r, W//r)
        x_un = self.unshuffle(x)

        # Step 2: move channels to last dimension for nn.Linear convenience
        # x_perm: (B, H', W', c_prime)
        x_perm = x_un.permute(0, 2, 3, 1)

        # Step 3: per-position projection -> (B, H', W', hidden_dim)
        hidden = self.proj_in(x_perm)

        # Step 4: RMSNorm then Hardsigmoid
        hidden = self.rmsnorm(hidden)
        hidden = self.act(hidden)

        # Step 5: global context pooling and gating
        # context: (B, hidden_dim)
        context = hidden.mean(dim=(1, 2))
        gate = self.context_proj(context)  # (B, hidden_dim)
        gate = self.act(gate)  # values in [0,1] approximately

        # Step 6: broadcast gate and modulate features
        gate = gate.unsqueeze(1).unsqueeze(1)  # (B,1,1,hidden_dim)
        hidden = hidden * gate  # (B, H', W', hidden_dim)

        # Step 7: project back to c_prime channels per position
        out_perm = self.proj_out(hidden)  # (B, H', W', c_prime)

        # Move channels back to channel-first layout and upscale spatially
        out_un = out_perm.permute(0, 3, 1, 2)  # (B, c_prime, H', W')
        out = self.shuffle(out_un)  # (B, in_channels, H, W)

        return out

# Configuration variables
batch_size = 8
in_channels = 3
height = 64
width = 64
downscale_factor = 2  # r
hidden_dim = 128

def get_inputs():
    """
    Returns a list containing a single input image tensor with shape
    (batch_size, in_channels, height, width).
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for the Model constructor:
    [in_channels, downscale_factor, hidden_dim]
    """
    return [in_channels, downscale_factor, hidden_dim]