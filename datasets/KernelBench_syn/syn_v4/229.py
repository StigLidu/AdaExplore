import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex example combining AdaptiveAvgPool2d, AdaptiveMaxPool1d, and RMSNorm.
    
    Computation pattern:
    1. Spatially downsample input with AdaptiveAvgPool2d to a fixed (pool_h, pool_w).
    2. Flatten the spatial dimensions into a 1D sequence per channel.
    3. Apply AdaptiveMaxPool1d to produce a fixed-length sequence (here equal to pool_h * pool_w).
    4. Permute and apply RMSNorm across channels for each sequence position.
    5. Aggregate global channel-wise context via mean over the sequence, pass through a small gating transform (Linear + tanh).
    6. Modulate the normalized sequence with the gating signal and reshape back to spatial form.
    7. Upsample back to the original spatial resolution using bilinear interpolation.
    """
    def __init__(self, channels: int, pool_h: int = 4, pool_w: int = 4, target_length: int = None, eps: float = 1e-8):
        """
        Initializes the composite module.

        Args:
            channels (int): Number of channels in the input tensor.
            pool_h (int): Target pooled height for AdaptiveAvgPool2d.
            pool_w (int): Target pooled width for AdaptiveAvgPool2d.
            target_length (int, optional): Output length for AdaptiveMaxPool1d. If None, uses pool_h * pool_w.
            eps (float, optional): Epsilon for RMSNorm stability.
        """
        super(Model, self).__init__()
        self.channels = channels
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.target_length = target_length if target_length is not None else (pool_h * pool_w)

        # Layers
        self.avg_pool = nn.AdaptiveAvgPool2d((self.pool_h, self.pool_w))
        # AdaptiveMaxPool1d expects input shape (B, C, L)
        self.adaptive_max = nn.AdaptiveMaxPool1d(self.target_length)
        # RMSNorm normalizes over the last dimension; we'll permute so last dim == channels
        self.rms = nn.RMSNorm(self.channels, eps=eps)
        # Simple gating transformation for channel-wise context
        self.gate_fc = nn.Linear(self.channels, self.channels)
        # Learnable residual scale for the output modulation (keeps same broadcast shape as (1, C, 1))
        self.res_scale = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor upsampled back to (batch_size, channels, height, width).
        """
        orig_h, orig_w = x.shape[2], x.shape[3]

        # 1) Spatial average pool to fixed small grid
        x_pooled = self.avg_pool(x)  # (B, C, pool_h, pool_w)

        B, C, H2, W2 = x_pooled.shape
        seq_len = H2 * W2

        # 2) Flatten spatial dims into a sequence per channel
        x_seq = x_pooled.view(B, C, seq_len)  # (B, C, S)

        # 3) Adaptive max pool over the sequence dimension to enforce fixed length
        x_maxpooled = self.adaptive_max(x_seq)  # (B, C, L) where L == target_length

        # 4) Permute so channels become the last dimension and apply RMSNorm over channels
        x_perm = x_maxpooled.permute(0, 2, 1)  # (B, L, C)
        x_norm = self.rms(x_perm)  # (B, L, C)
        x_norm = x_norm.permute(0, 2, 1)  # (B, C, L)

        # 5) Global channel-wise context (mean over sequence), gating transform
        global_ctx = x_norm.mean(dim=2)  # (B, C)
        gate = torch.tanh(self.gate_fc(global_ctx))  # (B, C)
        gate = gate.unsqueeze(2)  # (B, C, 1)

        # 6) Modulate normalized sequence with gating and residual scale
        out_seq = x_norm * (1.0 + gate) + self.res_scale  # (B, C, L)

        # 7) Reshape back to spatial grid and upsample to original resolution
        # Ensure target_length matches seq_len (pool_h * pool_w). If not, we adapt by interpolation in 1D.
        if out_seq.shape[2] != seq_len:
            # interpolate in sequence dimension to match spatial size
            out_seq = F.interpolate(out_seq, size=seq_len, mode='linear', align_corners=False)

        out_spatial = out_seq.view(B, C, H2, W2)  # (B, C, H2, W2)
        out_upsampled = F.interpolate(out_spatial, size=(orig_h, orig_w), mode='bilinear', align_corners=False)  # (B, C, H, W)

        return out_upsampled

# Module-level configuration variables
batch_size = 8
channels = 64
height = 128
width = 128
pool_h = 4
pool_w = 4
# target_length left as default (pool_h * pool_w)

def get_inputs():
    """
    Returns a list containing a single input tensor matching the configured sizes.
    """
    x = torch.randn(batch_size, channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters for Model: channels, pool_h, pool_w.
    """
    return [channels, pool_h, pool_w]