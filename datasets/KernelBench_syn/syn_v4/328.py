import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex module that combines 2D replication padding + unfolding-based patch projection,
    3D zero-padding with depth-reduction, spatial resizing, elementwise gating, and ReLU6.
    
    Computation steps:
    1. ReplicationPad2d on a 4D image tensor.
    2. Extract 3x3 patches via nn.Unfold and linearly project patch vectors.
    3. ZeroPad3d on a 5D volumetric tensor and reduce over depth using max.
    4. Spatially resize the depth-reduced map to match the projected 2D feature map.
    5. Use the resized single-channel spatial map as a gating mask applied to every channel.
    6. Apply ReLU6 and global average pooling to produce final per-batch feature vectors.
    """
    def __init__(self, in_channels: int, proj_channels: int, vol_channels: int):
        """
        Args:
            in_channels (int): Number of channels for the 2D input.
            proj_channels (int): Number of channels after projecting 3x3 patches.
            vol_channels (int): Number of channels in the 3D volumetric input.
        """
        super(Model, self).__init__()
        # Replication padding for the 2D input: (left, right, top, bottom)
        self.pad2d = nn.ReplicationPad2d((1, 2, 1, 2))
        # Extract 3x3 patches (no additional padding because we already padded)
        self.unfold = nn.Unfold(kernel_size=3, padding=0, stride=1)
        # Linear projection from flattened patch (C * 3 * 3) to proj_channels
        self.proj = nn.Linear(in_channels * 3 * 3, proj_channels, bias=True)
        # Zero pad 3D volumetric input: (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
        self.zpad3d = nn.ZeroPad3d((1, 1, 1, 1, 1, 1))
        # ReLU6 activation
        self.relu6 = nn.ReLU6()
        # store channels for reference
        self.in_channels = in_channels
        self.proj_channels = proj_channels
        self.vol_channels = vol_channels

    def forward(self, x2d: torch.Tensor, x3d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x2d (torch.Tensor): 4D tensor of shape (B, C_in, H, W).
            x3d (torch.Tensor): 5D tensor of shape (B, C_vol, D, H_vol, W_vol).

        Returns:
            torch.Tensor: Output tensor of shape (B, proj_channels) after spatial global pooling.
        """
        B, C, H, W = x2d.shape

        # 1) Pad the 2D input and extract patches
        x2d_p = self.pad2d(x2d)                       # (B, C, H+3, W+3)
        patches = self.unfold(x2d_p)                  # (B, C*9, L) where L = (H+1)*(W+1)
        patches_t = patches.transpose(1, 2)           # (B, L, C*9)

        # 2) Linear projection of each patch
        proj = self.proj(patches_t)                   # (B, L, proj_channels)

        # Recover spatial layout: L = H_out * W_out
        H_out = H + 1
        W_out = W + 1
        proj_map = proj.transpose(1, 2).contiguous()  # (B, proj_channels, L)
        proj_map = proj_map.view(B, self.proj_channels, H_out, W_out)  # (B, proj_channels, H_out, W_out)

        # 3) Zero-pad the 3D input and reduce over depth dimension
        x3d_p = self.zpad3d(x3d)                      # (B, C_vol, D+2, H_vol+2, W_vol+2)
        x3d_max = torch.max(x3d_p, dim=2)[0]          # (B, C_vol, H_vol+2, W_vol+2)

        # 4) Create a single-channel spatial gating map by averaging across channels
        gating_map = x3d_max.mean(dim=1, keepdim=True)  # (B, 1, Hg, Wg)

        # 5) Resize gating map to match projected 2D feature map spatial resolution
        gating_resized = F.interpolate(gating_map, size=(H_out, W_out), mode='bilinear', align_corners=False)
        # gating_resized: (B, 1, H_out, W_out)

        # 6) Apply gating (broadcast over channels), activation, and global pooling
        fused = proj_map * gating_resized            # (B, proj_channels, H_out, W_out)
        activated = self.relu6(fused)                 # (B, proj_channels, H_out, W_out)
        out = activated.mean(dim=(2, 3))              # Global average over spatial dims -> (B, proj_channels)

        return out

# Configuration / shape parameters
batch_size = 8
in_channels = 16
proj_channels = 32
vol_channels = 8
H = 32
W = 32
D = 5
H_vol = 16
W_vol = 16

def get_inputs():
    """
    Returns input tensors:
      - x2d: (batch_size, in_channels, H, W)
      - x3d: (batch_size, vol_channels, D, H_vol, W_vol)
    """
    x2d = torch.randn(batch_size, in_channels, H, W)
    x3d = torch.randn(batch_size, vol_channels, D, H_vol, W_vol)
    return [x2d, x3d]

def get_init_inputs():
    """
    Returns initialization arguments for the Model constructor:
      [in_channels, proj_channels, vol_channels]
    """
    return [in_channels, proj_channels, vol_channels]