import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex module combining a lazy transposed convolution (2D), lazy 3D instance normalization,
    and the Tanhshrink activation. The model upsamples a 2D feature map via ConvTranspose2d,
    merges it with a skip-connection tensor, applies 3D InstanceNorm (after adding a singleton
    depth dimension), uses Tanhshrink, and finally adds a resized residual from the original input.
    """
    def __init__(self, deconv_out_channels: int = 16, deconv_kernel: int = 3, deconv_stride: int = 2):
        super(Model, self).__init__()
        # LazyConvTranspose2d will infer in_channels at first forward pass
        # We set out_channels explicitly to control resulting channel width.
        self.deconv = nn.LazyConvTranspose2d(
            out_channels=deconv_out_channels,
            kernel_size=deconv_kernel,
            stride=deconv_stride,
            padding=1,
            output_padding=1,
            bias=True,
        )
        # LazyInstanceNorm3d will infer num_features when given a 5D tensor
        self.inst_norm3d = nn.LazyInstanceNorm3d()
        # Element-wise activation
        self.act = nn.Tanhshrink()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
          1. Upsample x with a transposed convolution -> u (B, C_out, H_out, W_out)
          2. Resize skip to match (H_out, W_out), then add singleton depth dim -> s5 (B, C_skip, 1, H_out, W_out)
          3. Unsqueeze u to 5D -> u5 (B, C_out, 1, H_out, W_out)
          4. Concatenate along channel dimension -> z5
          5. Apply LazyInstanceNorm3d -> n5
          6. Squeeze depth -> n4, apply Tanhshrink -> a4
          7. Add a resized residual from x -> out

        Args:
            x: Input 4D tensor (B, C_in, H, W)
            skip: Skip connection 4D tensor (B, C_skip, H_skip, W_skip)

        Returns:
            out: Output 4D tensor (B, C_out_combined, H_out, W_out)
        """
        # 1) Transposed convolution -> upsample spatially
        u = self.deconv(x)  # (B, C_out, H_out, W_out)

        # 2) Resize skip to match u's spatial dims
        u_h, u_w = u.shape[2], u.shape[3]
        # Use bilinear interpolation for skip (assumes 4D)
        s_resized = F.interpolate(skip, size=(u_h, u_w), mode='bilinear', align_corners=False)

        # 3) Add singleton depth dimension to both tensors to make them 5D for InstanceNorm3d
        u5 = u.unsqueeze(2)         # (B, C_out, 1, H_out, W_out)
        s5 = s_resized.unsqueeze(2) # (B, C_skip, 1, H_out, W_out)

        # 4) Concatenate along channels
        z5 = torch.cat([u5, s5], dim=1)  # (B, C_out + C_skip, 1, H_out, W_out)

        # 5) Apply lazy 3D instance normalization
        n5 = self.inst_norm3d(z5)  # (B, C_out + C_skip, 1, H_out, W_out)

        # 6) Remove depth dim and apply Tanhshrink
        n4 = n5.squeeze(2)  # (B, C_out + C_skip, H_out, W_out)
        a4 = self.act(n4)

        # 7) Add a resized residual connection from original x (projected spatially to match)
        res = F.interpolate(x, size=(u_h, u_w), mode='bilinear', align_corners=False)

        # If residual channels differ, project residual to match channel count via a simple convolution.
        # Use a lightweight 1x1 conv created on the fly to avoid adding persistent parameters.
        if res.shape[1] != a4.shape[1]:
            # Create a temporary conv with appropriate in/out channels; initialize to identity-like behavior
            conv1x1 = nn.Conv2d(res.shape[1], a4.shape[1], kernel_size=1, bias=False).to(res.device)
            # Initialize weights to small values so it doesn't dominate initially
            nn.init.kaiming_uniform_(conv1x1.weight, a=0.01)
            res_proj = conv1x1(res)
        else:
            res_proj = res

        out = a4 + res_proj
        return out

# Configuration variables
BATCH = 2
IN_CHANNELS = 3
SKIP_CHANNELS = 8
H = 31
W = 45

def get_inputs():
    """
    Returns example inputs:
      - x: (BATCH, IN_CHANNELS, H, W)
      - skip: (BATCH, SKIP_CHANNELS, H_skip, W_skip) with potentially different spatial dims
    """
    x = torch.randn(BATCH, IN_CHANNELS, H, W)
    # Provide skip with different spatial size to exercise interpolation logic
    skip = torch.randn(BATCH, SKIP_CHANNELS, max(1, H // 2), max(1, W // 3))
    return [x, skip]

def get_init_inputs():
    """
    No special initialization inputs required; layers are lazily initialized on first forward.
    """
    return []