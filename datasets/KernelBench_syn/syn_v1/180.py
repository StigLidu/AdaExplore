import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Multi-scale upsampling fusion module that:
    - Upsamples an input tensor with two different upsampling modules (nn.UpsamplingBilinear2d and nn.Upsample).
    - Applies Layer Normalization to the first upsampled feature map.
    - Computes a spatial gate from the normalized features and uses it to modulate the second upsampled branch.
    - Concatenates the modulated second branch with a resized copy of the normalized first branch and applies a second LayerNorm.

    The design demonstrates combining vision upsampling layers and LayerNorm with tensor reshaping, interpolation,
    gating, and channel fusion.
    """
    def __init__(self, in_channels: int, in_h: int, in_w: int, scale_factor1: int, scale_factor2: int):
        """
        Args:
            in_channels: Number of input channels.
            in_h: Input height.
            in_w: Input width.
            scale_factor1: Integer scale factor for nn.UpsamplingBilinear2d (first branch).
            scale_factor2: Integer scale factor for nn.Upsample (second branch).
        """
        super(Model, self).__init__()
        assert isinstance(scale_factor1, int) and isinstance(scale_factor2, int), "Scale factors must be integers."

        self.in_channels = in_channels
        self.in_h = in_h
        self.in_w = in_w
        self.scale_factor1 = scale_factor1
        self.scale_factor2 = scale_factor2

        # Compute target spatial sizes for both branches
        self.h1 = in_h * scale_factor1
        self.w1 = in_w * scale_factor1
        self.h2 = in_h * scale_factor2
        self.w2 = in_w * scale_factor2

        # First branch: older UpSampling module (bilinear)
        self.up_bilinear = nn.UpsamplingBilinear2d(scale_factor=scale_factor1)

        # LayerNorm that normalizes over (C, H1, W1) for the first upsampled tensor
        self.ln1 = nn.LayerNorm((in_channels, self.h1, self.w1))

        # Second branch: generic Upsample with bilinear interpolation (different scale)
        self.up_generic = nn.Upsample(scale_factor=scale_factor2, mode='bilinear', align_corners=True)

        # After concatenation, channels double -> normalize over (2*C, H2, W2)
        self.ln2 = nn.LayerNorm((2 * in_channels, self.h2, self.w2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with the following steps:
        1. Upsample x with nn.UpsamplingBilinear2d -> x1 (B, C, H1, W1)
        2. Apply LayerNorm over (C, H1, W1) -> x1_norm
        3. Upsample x with nn.Upsample -> x2 (B, C, H2, W2)
        4. Compute a spatial gate from x1_norm: mean over channels -> (B,1,H1,W1), then sigmoid -> gate
        5. Resize gate to x2 spatial resolution and modulate x2
        6. Resize x1_norm to x2 resolution and concatenate with modulated x2 -> (B, 2C, H2, W2)
        7. Apply LayerNorm over (2C, H2, W2) and return
        """
        # Branch A: bilinear upsampling + LayerNorm
        x1 = self.up_bilinear(x)                      # (B, C, H1, W1)
        x1_norm = self.ln1(x1)                        # (B, C, H1, W1)

        # Branch B: generic upsample to a different scale
        x2 = self.up_generic(x)                       # (B, C, H2, W2)

        # Spatial gating: collapse channels of x1_norm to produce a single-channel spatial map
        gate = torch.mean(x1_norm, dim=1, keepdim=True)   # (B, 1, H1, W1)
        gate = torch.sigmoid(gate)                        # (B, 1, H1, W1)

        # Resize gate to match x2 spatial resolution (H2, W2)
        gate_resized = F.interpolate(gate, size=(self.h2, self.w2), mode='bilinear', align_corners=True)  # (B,1,H2,W2)

        # Modulate x2 with the resized gate (broadcast over channels)
        x2_mod = x2 * gate_resized                       # (B, C, H2, W2)

        # Resize normalized x1 to x2 resolution to fuse both branches
        x1_resized = F.interpolate(x1_norm, size=(self.h2, self.w2), mode='bilinear', align_corners=True)  # (B, C, H2, W2)

        # Concatenate along channel dimension and normalize
        fused = torch.cat([x2_mod, x1_resized], dim=1)   # (B, 2C, H2, W2)
        out = self.ln2(fused)                            # (B, 2C, H2, W2)

        return out

# Configuration / initialization parameters at module level
batch_size = 4
in_channels = 3
in_h = 64
in_w = 48
scale_factor1 = 2  # used by nn.UpsamplingBilinear2d
scale_factor2 = 3  # used by nn.Upsample

def get_inputs():
    """
    Returns a list containing a single input tensor with shape:
    (batch_size, in_channels, in_h, in_w)
    """
    x = torch.randn(batch_size, in_channels, in_h, in_w)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters for the Model constructor in order:
    [in_channels, in_h, in_w, scale_factor1, scale_factor2]
    """
    return [in_channels, in_h, in_w, scale_factor1, scale_factor2]