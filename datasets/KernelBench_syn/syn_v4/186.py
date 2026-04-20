import torch
import torch.nn as nn

# Configuration variables
BATCH_SIZE = 8
IN_CHANNELS = 32
DEPTH = 16
HEIGHT = 32
WIDTH = 32
PAD = 1  # symmetric pad for ReflectionPad3d
POOL_OD = 4
POOL_OH = 8
POOL_OW = 8

class Model(nn.Module):
    """
    Complex 3D feature aggregation module that:
      - Pads the input volumetric tensor using reflection padding
      - Applies adaptive max pooling to a target 3D spatial size
      - Projects channel features via a learned linear mapping (per spatial location)
      - Builds a global context vector by averaging projected features across spatial locations
      - Computes spatial attention scores via einsum between context and per-location features
      - Normalizes the spatial attention with LogSoftmax, applies it, aggregates weighted features
      - Produces final channel logits and returns channel-wise log-probabilities via LogSoftmax

    This module demonstrates a blend of ReflectionPad3d, AdaptiveMaxPool3d and LogSoftmax,
    combined with tensor rearrangements and einsum-based attention computations.

    Args:
        in_channels (int): Number of input channels C.
        out_d (int): Output pooled depth.
        out_h (int): Output pooled height.
        out_w (int): Output pooled width.
        pad (int or tuple): Padding size for ReflectionPad3d.
    """
    def __init__(self, in_channels: int, out_d: int, out_h: int, out_w: int, pad=1):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.pad_layer = nn.ReflectionPad3d(pad)
        self.pool = nn.AdaptiveMaxPool3d((out_d, out_h, out_w))
        # Linear projection across channels applied per spatial location:
        # we'll map C -> C to mix channel information locally
        self.channel_proj = nn.Linear(in_channels, in_channels, bias=False)
        # Final linear to produce channel logits (can be used for classification or scoring)
        self.final_linear = nn.Linear(in_channels, in_channels, bias=True)
        # LogSoftmax over spatial positions and over channels
        self.logsoftmax_spatial = nn.LogSoftmax(dim=1)  # normalize over S (flattened spatial positions)
        self.logsoftmax_channel = nn.LogSoftmax(dim=1)  # normalize over channel dimension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, D, H, W).

        Returns:
            torch.Tensor: Log-probabilities over channels of shape (B, C).
        """
        # 1) Reflection padding
        x_padded = self.pad_layer(x)  # -> (B, C, D + pad*2, H + pad*2, W + pad*2)

        # 2) Adaptive max pool to desired output spatial shape
        x_pool = self.pool(x_padded)  # -> (B, C, od, oh, ow)
        B, C, od, oh, ow = x_pool.shape
        S = od * oh * ow  # flattened spatial size

        # 3) Rearrange to (B, S, C) so we can apply a channel-wise linear projection per location
        x_feat = x_pool.permute(0, 2, 3, 4, 1).contiguous().view(B, S, C)  # (B, S, C)

        # 4) Channel projection (learned mixing of channels per spatial location)
        x_proj = self.channel_proj(x_feat)  # (B, S, C)

        # 5) Global context vector: mean across spatial positions -> (B, C)
        context = x_proj.mean(dim=1)  # (B, C)

        # 6) Attention scores: einsum between context and per-location features -> (B, S)
        att_logits = torch.einsum('bc,bsc->bs', context, x_proj)  # (B, S)

        # 7) Normalize attention over spatial locations with LogSoftmax, then exponentiate to get weights
        att_logprob = self.logsoftmax_spatial(att_logits)  # (B, S)
        att_weights = att_logprob.exp()  # (B, S)

        # 8) Weighted aggregation of projected features -> (B, C)
        weighted = torch.einsum('bs,bsc->bc', att_weights, x_proj)  # (B, C)

        # 9) Final linear mapping and channel-wise LogSoftmax output
        channel_logits = self.final_linear(weighted)  # (B, C)
        out_log_probs = self.logsoftmax_channel(channel_logits)  # (B, C)

        return out_log_probs

# Test configuration for creating inputs
B = BATCH_SIZE
C = IN_CHANNELS
D = DEPTH
H = HEIGHT
W = WIDTH
pad = PAD
od = POOL_OD
oh = POOL_OH
ow = POOL_OW

def get_inputs():
    """
    Returns example input tensors for the module's forward pass:
      - x: random volumetric tensor shaped (B, C, D, H, W)
    """
    x = torch.randn(B, C, D, H, W)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters for the Model:
      [in_channels, out_d, out_h, out_w, pad]
    """
    return [C, od, oh, ow, pad]