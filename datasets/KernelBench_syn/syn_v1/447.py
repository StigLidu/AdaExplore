import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex 3D-to-2D processing module that demonstrates:
      - 3D reflection padding
      - Adaptive 3D max pooling
      - Lazy transposed 2D convolution applied after reshaping a 3D tensor
      - Batch normalization and ReLU non-linearity
      - A final adaptive 3D pooling followed by depth-wise reduction to produce a 2D feature map

    Forward pass summary:
      1. Input: (B, C, D, H, W)
      2. Reflection-pad in 3D
      3. AdaptiveMaxPool3d -> (B, C, D1, H1, W1)
      4. Permute & reshape to collapse depth into batch dimension -> 4D tensor
      5. LazyConvTranspose2d (learns in_channels at first forward) + BatchNorm2d + ReLU
      6. Reshape back to 5D (B, outC, D1, H2, W2)
      7. AdaptiveMaxPool3d to reduce to (B, outC, D2, H3, W3)
      8. Reduce over depth (mean) -> final (B, outC, H3, W3)
    """
    def __init__(self, out_channels: int = 16):
        super(Model, self).__init__()
        # ReflectionPad3d: pads (left, right, top, bottom, front, back)
        self.pad3d = nn.ReflectionPad3d((1, 1, 2, 2, 1, 1))

        # First adaptive pooling reduces resolution in depth/height/width
        # to manageable sizes for the subsequent reshape & conv-transpose.
        self.pool1 = nn.AdaptiveMaxPool3d((8, 16, 16))

        # LazyConvTranspose2d will infer in_channels on the first forward call.
        # It upsamples spatial dimensions by a factor ~2 (stride=2).
        self.deconv2d = nn.LazyConvTranspose2d(out_channels=out_channels,
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1)

        # BatchNorm2d over the output channels of the transposed conv
        self.bn2d = nn.BatchNorm2d(out_channels)

        # Second adaptive pooling compresses depth and spatial dims to final targets
        self.pool2 = nn.AdaptiveMaxPool3d((4, 8, 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 5D tensor with shape (B, C, D, H, W)

        Returns:
            torch.Tensor: 4D tensor (B, out_channels, H_out, W_out)
        """
        if x.dim() != 5:
            raise ValueError("Input must be a 5D tensor (B, C, D, H, W)")

        # 1) Reflection pad in 3D
        x = self.pad3d(x)

        # 2) Reduce resolution with adaptive max pooling
        x = self.pool1(x)  # shape: (B, C, D1, H1, W1)
        B, C, D1, H1, W1 = x.shape

        # 3) Collapse depth into batch dimension to apply 2D transposed conv independently per depth slice
        # Permute to (B, D1, C, H1, W1) then reshape to (B*D1, C, H1, W1)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x_4d = x.view(B * D1, C, H1, W1)

        # 4) Apply LazyConvTranspose2d -> BatchNorm2d -> ReLU
        x_4d = self.deconv2d(x_4d)           # shape: (B*D1, outC, H2, W2)
        x_4d = self.bn2d(x_4d)
        x_4d = F.relu(x_4d)

        # 5) Reshape back to 5D: (B, D1, outC, H2, W2) -> permute to (B, outC, D1, H2, W2)
        outC = x_4d.shape[1]
        H2, W2 = x_4d.shape[2], x_4d.shape[3]
        x_5d = x_4d.view(B, D1, outC, H2, W2).permute(0, 2, 1, 3, 4).contiguous()

        # 6) Final adaptive 3D pooling to compress to desired smaller depth+spatial dims
        x_pooled = self.pool2(x_5d)  # shape: (B, outC, D2, H3, W3)

        # 7) Reduce over the depth dimension to produce a 2D feature map per sample
        out = x_pooled.mean(dim=2)   # shape: (B, outC, H3, W3)

        return out


# Configuration / default sizes for generating inputs
batch_size = 8
in_channels = 3
depth = 16
height = 64
width = 64

def get_inputs():
    """
    Produces the input tensor matching the model expectations:
    A random 5D tensor of shape (batch_size, in_channels, depth, height, width)
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    No extra initialization inputs required. LazyConvTranspose2d will infer
    its in_channels on the first forward pass from the input tensor.
    """
    return []