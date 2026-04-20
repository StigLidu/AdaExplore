import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex 3D -> 2D hybrid processing model.

    Pipeline:
    - 3D convolution to produce intermediate feature map (B, C1, D, H, W)
    - ReLU activation
    - Treat each depth slice as an independent 2D sample:
        reshape (B, C1, D, H, W) -> (B*D, C1, H, W)
      then apply InstanceNorm2d and LPPool2d (reduces H/W).
    - Restore depth dimension -> (B, C1, D, H2, W2)
    - 1x1x1 Conv3d to mix channels -> (B, C2, D, H2, W2)
    - Project original input (residual) to match channels and downsample spatially using adaptive pooling
    - Add residual, final ReLU and return final 5D tensor
    """
    def __init__(
        self,
        in_channels: int,
        conv1_out: int,
        conv2_out: int,
        lp_p: int = 2,
        lp_kernel: int = 3,
        inst_eps: float = 1e-5,
        inst_affine: bool = True
    ):
        """
        Args:
            in_channels: number of channels in input (C)
            conv1_out: number of filters for first Conv3d
            conv2_out: number of filters for final Conv3d
            lp_p: p parameter for LPPool2d
            lp_kernel: kernel size for LPPool2d
            inst_eps: epsilon for InstanceNorm2d
            inst_affine: whether InstanceNorm2d has affine parameters
        """
        super(Model, self).__init__()

        # 3D conv extractor preserves spatial dims (padding=1 for kernel 3)
        self.conv3d_1 = nn.Conv3d(in_channels, conv1_out, kernel_size=(3, 3, 3), padding=1)
        # 1x1x1 conv to remap to final channel dimension after 2D pooling
        self.conv3d_2 = nn.Conv3d(conv1_out, conv2_out, kernel_size=(1, 1, 1))
        # Projection for residual connection when channel dims differ
        self.proj_residual = nn.Conv3d(in_channels, conv2_out, kernel_size=1)
        # 2D InstanceNorm applied to each depth slice separately
        self.inst_norm_2d = nn.InstanceNorm2d(conv1_out, eps=inst_eps, affine=inst_affine)
        # LPPool2d reduces spatial H/W for each (B*D) sample
        self.lp_pool = nn.LPPool2d(norm_type=lp_p, kernel_size=lp_kernel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (B, C_in, D, H, W)

        Returns:
            tensor of shape (B, C_out, D, H2, W2) where H2/W2 are reduced by LPPool2d
        """
        # Save residual for later
        residual = x  # (B, C_in, D, H, W)

        # 1) 3D conv + activation
        out = self.conv3d_1(x)   # (B, C1, D, H, W)
        out = self.relu(out)

        # 2) Prepare for 2D operations: treat each depth slice as an independent 2D sample
        B, C1, D, H, W = out.shape
        # Permute to (B, D, C1, H, W) then collapse to (B*D, C1, H, W)
        out_2d = out.permute(0, 2, 1, 3, 4).contiguous().view(B * D, C1, H, W)

        # 3) InstanceNorm2d (per-sample) and LPPool2d (reduces H/W)
        out_2d = self.inst_norm_2d(out_2d)  # (B*D, C1, H, W)
        out_2d = self.lp_pool(out_2d)       # (B*D, C1, H2, W2)

        # 4) Restore to 5D: (B, C1, D, H2, W2)
        _, _, H2, W2 = out_2d.shape
        out = out_2d.view(B, D, C1, H2, W2).permute(0, 2, 1, 3, 4).contiguous()

        # 5) 1x1x1 Conv3d to mix channels -> (B, C2, D, H2, W2)
        out = self.conv3d_2(out)

        # 6) Prepare residual: downsample spatial dims to match H2/W2 using adaptive pooling
        res_down = F.adaptive_avg_pool3d(residual, (out.shape[2], out.shape[3], out.shape[4]))  # (B, C_in, D, H2, W2)
        res_proj = self.proj_residual(res_down)  # (B, C2, D, H2, W2)

        # 7) Merge with residual and apply final activation
        out = self.relu(out + res_proj)

        return out


# Configuration variables (module level)
batch_size = 2
in_channels = 3
depth = 8
height = 128
width = 128

conv1_out = 16
conv2_out = 32

lp_p = 2         # Lp norm type for LPPool2d
lp_kernel = 3    # kernel size for LPPool2d

def get_inputs():
    """
    Returns:
        A single input tensor shaped (B, C_in, D, H, W)
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization arguments for Model(...) in the same order as the constructor:
        (in_channels, conv1_out, conv2_out, lp_p, lp_kernel)
    """
    return [in_channels, conv1_out, conv2_out, lp_p, lp_kernel]