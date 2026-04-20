import torch
import torch.nn as nn
from typing import List, Tuple

"""
Complex module combining ConstantPad2d, AdaptiveAvgPool2d, and Softmax to form
a small channel-attention + spatial pooling block. The module:
 - Pads the input with a constant value
 - Computes channel-wise attention from globally pooled context (Softmax over channels)
 - Modulates the padded input by the attention
 - Produces a compact two-channel spatial summary by applying an adaptive spatial
   pooling to the channel-averaged feature map and concatenating a global summary
"""

# Configuration (module-level)
batch_size = 8
channels = 12
height = 32
width = 48

# Padding specified as (left, right, top, bottom)
pad_left = 2
pad_right = 3
pad_top = 1
pad_bottom = 0

pad_value = 0.1  # constant pad value
out_h = 8        # spatial output height after adaptive pooling
out_w = 12       # spatial output width after adaptive pooling

class Model(nn.Module):
    """
    Channel-attention spatial summarizer.

    Args:
        padding (Tuple[int, int, int, int]): Padding tuple (left, right, top, bottom).
        pad_value (float): Constant value to use for padding.
        out_h (int): Output height for the spatial adaptive average pooling.
        out_w (int): Output width for the spatial adaptive average pooling.
    """
    def __init__(self, padding: Tuple[int, int, int, int], pad_value: float, out_h: int, out_w: int):
        super(Model, self).__init__()
        # Padding layer: pads (left, right, top, bottom)
        self.pad = nn.ConstantPad2d(padding, pad_value)
        # Global channel context (to produce channel descriptors)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Spatial downsampling for the channel-averaged map
        self.spatial_pool = nn.AdaptiveAvgPool2d((out_h, out_w))
        # Softmax to convert channel descriptors to attention weights
        self.softmax = nn.Softmax(dim=1)
        # Learnable temperature/scale for attention logits
        self.attn_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the module.

        Steps:
         1. Pad input with a constant value.
         2. Compute per-channel global descriptors via AdaptiveAvgPool2d -> (B, C, 1, 1).
         3. Squeeze descriptors and apply scaled Softmax across channels to get attention (B, C).
         4. Reshape attention to (B, C, 1, 1) and modulate the padded input.
         5. Compute channel-average of modulated feature map -> (B, 1, H_pad, W_pad).
         6. Apply spatial adaptive pooling to get downsampled map (B, 1, out_h, out_w).
         7. Also compute a compact global summary (sum of channel descriptors) and expand
            to match spatial dims, then concatenate with the downsampled map along channel dim.
        """
        # 1) Pad
        padded = self.pad(x)  # shape: (B, C, H_pad, W_pad)

        # 2) Global pooling -> channel descriptors
        desc = self.global_pool(padded)  # (B, C, 1, 1)

        # 3) Squeeze and apply scaled softmax across channels
        B, C, _, _ = desc.shape
        desc_flat = desc.view(B, C)                 # (B, C)
        attn_logits = desc_flat * self.attn_scale   # scale logits
        attn = self.softmax(attn_logits)            # (B, C) sums to 1 across channels

        # 4) Reshape attention and modulate padded input
        attn_map = attn.view(B, C, 1, 1)            # (B, C, 1, 1)
        modulated = padded * attn_map               # (B, C, H_pad, W_pad)

        # 5) Channel-average to get a single-channel spatial map
        spatial_map = modulated.mean(dim=1, keepdim=True)  # (B, 1, H_pad, W_pad)

        # 6) Spatial adaptive pooling to a compact 2D map
        down = self.spatial_pool(spatial_map)       # (B, 1, out_h, out_w)

        # 7) Global summary: sum of descriptors across channels -> (B, 1, 1, 1), expand to spatial
        global_summary = desc_flat.sum(dim=1, keepdim=True).view(B, 1, 1, 1)  # (B,1,1,1)
        global_expanded = global_summary.expand(-1, -1, down.shape[2], down.shape[3])  # (B,1,out_h,out_w)

        # Concatenate downsampled spatial map with the expanded global summary -> (B, 2, out_h, out_w)
        out = torch.cat([down, global_expanded], dim=1)

        return out

def get_inputs() -> List[torch.Tensor]:
    """
    Returns a list containing a single input tensor with the configured batch size,
    number of channels, height and width.
    """
    x = torch.randn(batch_size, channels, height, width)
    return [x]

def get_init_inputs() -> List:
    """
    Returns initialization parameters required by the Model constructor.
    Order: padding tuple (left, right, top, bottom), pad_value, out_h, out_w
    """
    padding = (pad_left, pad_right, pad_top, pad_bottom)
    return [padding, pad_value, out_h, out_w]