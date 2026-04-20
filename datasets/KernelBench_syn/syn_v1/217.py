import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex example combining LazyInstanceNorm3d, spatial upsampling (nearest neighbor)
    and LocalResponseNorm. The computation pipeline is as follows:

    1. Apply LazyInstanceNorm3d to a 5D input tensor (B, C, D, H, W).
       This normalizes each instance over spatial dims and channels once channels are known.
    2. Collapse the depth dimension by computing the mean across D -> (B, C, H, W).
    3. Upsample the resulting 4D tensor using nearest-neighbor interpolation.
    4. Apply LocalResponseNorm on the upsampled 4D tensor to introduce cross-channel normalization.
    5. Broadcast the normalized upsampled tensor back to a 5D tensor and use it to modulate
       the original instance-normalized input that has been resized to the same spatial resolution.
    6. Reduce across channels to produce a compact 4D output (B, D, H_up, W_up).

    This demonstrates reshaping/reshuffling between 4D/5D tensors, module reuse (upsampling applied
    to both collapsed and per-slice tensors), and integration of three provided layers.
    """
    def __init__(self, upsample_scale: int = 2, lrn_size: int = 5, lrn_alpha: float = 1e-4, lrn_beta: float = 0.75, lrn_k: float = 1.0):
        """
        Args:
            upsample_scale: Spatial upsampling scale factor for height and width.
            lrn_size: LocalResponseNorm window size (number of channels to sum over).
            lrn_alpha: LocalResponseNorm alpha parameter.
            lrn_beta: LocalResponseNorm beta parameter.
            lrn_k: LocalResponseNorm k parameter.
        """
        super(Model, self).__init__()
        # LazyInstanceNorm3d will infer num_features (channels) on the first forward pass.
        self.inst_norm = nn.LazyInstanceNorm3d()  # normalization over (C, D, H, W) per instance
        # Upsample module for 2D nearest-neighbor upsampling of (N, C, H, W) tensors.
        self.upsample = nn.UpsamplingNearest2d(scale_factor=upsample_scale)
        # Local response normalization applied after upsampling on the collapsed depth representation.
        self.lrn = nn.LocalResponseNorm(size=lrn_size, alpha=lrn_alpha, beta=lrn_beta, k=lrn_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, D, H, W)

        Returns:
            Tensor of shape (B, D, H_up, W_up) where H_up = H * scale, W_up = W * scale.
            The result is a channel-collapsed, depth-preserving spatial feature map that has been
            modulated by local response normalization derived from the depth-collapsed statistics.
        """
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape

        # 1) Instance normalization on the 3D volume per instance (lazy init will set C)
        x_norm = self.inst_norm(x)  # (B, C, D, H, W)

        # 2) Collapse depth to create a 4D spatial summary by averaging over depth
        depth_mean = x_norm.mean(dim=2)  # (B, C, H, W)

        # 3) Upsample the collapsed depth summary in spatial dims (H, W) -> (H_up, W_up)
        upsampled = self.upsample(depth_mean)  # (B, C, H_up, W_up)

        # 4) Apply Local Response Normalization to the upsampled summary
        lrn_out = self.lrn(upsampled)  # (B, C, H_up, W_up)

        # 5) Prepare to modulate the original per-slice normalized input:
        #    - Resize the original normalized per-slice features to the upsampled spatial size.
        #      We reshape (B, C, D, H, W) -> (B*D, C, H, W), upsample, then reshape back.
        x_per_slice = x_norm.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)  # (B*D, C, H, W)
        x_per_slice_up = self.upsample(x_per_slice)  # (B*D, C, H_up, W_up)
        x_up = x_per_slice_up.reshape(B, D, C, x_per_slice_up.shape[-2], x_per_slice_up.shape[-1]).permute(0, 2, 1, 3, 4)
        # x_up shape: (B, C, D, H_up, W_up)

        # 6) Broadcast the LRN output back to 5D to match x_up and modulate
        lrn_rep = lrn_out.unsqueeze(2).expand(-1, -1, D, -1, -1)  # (B, C, D, H_up, W_up)
        gated = x_up * torch.sigmoid(lrn_rep)  # element-wise gating modulation

        # 7) Reduce across channels to produce a compact output (e.g., channel summation)
        out = gated.sum(dim=1)  # (B, D, H_up, W_up)

        return out

# Configuration variables
BATCH = 8
CHANNELS = 32  # used implicitly by LazyInstanceNorm3d on first forward
DEPTH = 16
HEIGHT = 64
WIDTH = 64
UPSCALE = 2

def get_inputs():
    """
    Generates a random 5D tensor for testing: (BATCH, CHANNELS, DEPTH, HEIGHT, WIDTH).
    Uses normal distribution to simulate feature maps / volumetric data.
    """
    x = torch.randn(BATCH, CHANNELS, DEPTH, HEIGHT, WIDTH)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for the Model constructor. In this example we provide
    the upsample scale and LocalResponseNorm parameters (size, alpha, beta, k) to demonstrate
    get_init_inputs usage, although the model also works with its defaults.
    """
    # (upsample_scale, lrn_size, lrn_alpha, lrn_beta, lrn_k)
    return [UPSCALE, 5, 1e-4, 0.75, 1.0]