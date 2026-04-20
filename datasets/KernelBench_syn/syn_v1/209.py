import torch
import torch.nn as nn

# Configuration variables
BATCH = 4
C_IN = 8
C_MID = 24
C_OUT = 12
D = 8
H = 64
W = 64

# ConvTranspose3d parameters
KERNEL = 3
STRIDE = 2
PADDING = 1
OUTPUT_PADDING = 1

# AvgPool2d parameter
POOL_KERNEL = 2


class Model(nn.Module):
    """
    Complex 3D-to-2D hybrid model demonstrating:
      - 3D transposed convolution to upsample volumetric data
      - 3D instance normalization
      - Non-linearity
      - Collapsing the depth axis to form 2D feature maps
      - 2D average pooling to reduce spatial resolution
      - Final 3D transposed convolution applied to a single-depth volume

    Forward pipeline:
      x (N, C_IN, D, H, W)
        -> ConvTranspose3d -> (N, C_MID, D2, H2, W2)
        -> InstanceNorm3d -> ReLU
        -> mean over depth -> (N, C_MID, H2, W2)
        -> AvgPool2d -> (N, C_MID, H3, W3)
        -> unsqueeze depth=1 -> (N, C_MID, 1, H3, W3)
        -> final ConvTranspose3d (1x1x1) -> (N, C_OUT, D_out, H3, W3)
    """
    def __init__(self):
        super(Model, self).__init__()
        # Upsample volumetric input
        self.upconv = nn.ConvTranspose3d(
            in_channels=C_IN,
            out_channels=C_MID,
            kernel_size=KERNEL,
            stride=STRIDE,
            padding=PADDING,
            output_padding=OUTPUT_PADDING,
            bias=True
        )

        # Normalize across each instance for the 3D feature maps
        self.inst_norm = nn.InstanceNorm3d(num_features=C_MID, affine=True)

        # Final 1x1x1 transposed convolution to map back to desired output channels
        # It expects a 5D input; we'll feed it a single-depth volume after pooling.
        self.final_conv = nn.ConvTranspose3d(
            in_channels=C_MID,
            out_channels=C_OUT,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            bias=True
        )

        # 2D average pooling for spatial downsampling after collapsing depth
        self.avgpool2d = nn.AvgPool2d(kernel_size=POOL_KERNEL)

        # Non-linearity
        self.act = nn.ReLU(inplace=True)

        # Initialize weights with a small normal std for stability
        nn.init.normal_(self.upconv.weight, mean=0.0, std=0.02)
        if self.upconv.bias is not None:
            nn.init.zeros_(self.upconv.bias)
        nn.init.normal_(self.final_conv.weight, mean=0.0, std=0.02)
        if self.final_conv.bias is not None:
            nn.init.zeros_(self.final_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input volumetric tensor of shape (N, C_IN, D, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (N, C_OUT, D_out, H_out, W_out)
                          where D_out is typically 1 (from unsqueezing) and
                          H_out/W_out depend on pooling and upsampling.
        """
        # 1) Upsample volumetric features with transposed conv
        y = self.upconv(x)  # (N, C_MID, D2, H2, W2)

        # 2) Instance normalization across channels per sample
        y = self.inst_norm(y)

        # 3) Non-linearity
        y = self.act(y)

        # 4) Collapse the depth dimension by averaging to obtain 2D feature maps
        y2d = torch.mean(y, dim=2)  # (N, C_MID, H2, W2)

        # 5) Spatial average pooling to reduce H/W
        y2d = self.avgpool2d(y2d)  # (N, C_MID, H3, W3)

        # 6) Convert back to a single-depth 5D tensor to apply final 3D conv
        y3d = y2d.unsqueeze(2)  # (N, C_MID, 1, H3, W3)

        # 7) Final 1x1x1 transposed convolution to project to output channels
        out = self.final_conv(y3d)  # (N, C_OUT, D_out=1, H3, W3)

        return out


def get_inputs():
    """
    Returns:
        List[torch.Tensor]: [x] where x has shape (BATCH, C_IN, D, H, W)
    """
    x = torch.randn(BATCH, C_IN, D, H, W)
    return [x]


def get_init_inputs():
    """
    No additional initialization parameters required for this module.
    """
    return []