import torch
import torch.nn as nn
from typing import List, Tuple

class Model(nn.Module):
    """
    Complex 3D-to-2D feature extractor and classifier that demonstrates:
      - Replication padding in 3D (nn.ReplicationPad3d)
      - Two 3D convolutions with non-linearity
      - RMS-based normalization applied to channel dimension (nn.RMSNorm)
      - Depth-wise collapse to 2D followed by lazy BatchNorm2d (nn.LazyBatchNorm2d)
      - Global spatial pooling and final linear classification head

    The model accepts a 5D input tensor: (batch, channels, depth, height, width)
    and produces class logits of shape (batch, num_classes).
    """
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_size: int,
        pad_sizes: Tuple[int, int, int, int, int, int],
        dilation: int,
        num_classes: int,
    ):
        super(Model, self).__init__()
        # 3D replication padding: (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
        self.pad3d = nn.ReplicationPad3d(pad_sizes)

        # First 3D conv - we handle padding explicitly with ReplicationPad3d
        self.conv3d_1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,       # padding handled by self.pad3d
            dilation=dilation,
            bias=True,
        )

        # Activation
        self.act = nn.GELU()

        # Second 3D conv acts as a channel projector
        self.conv3d_2 = nn.Conv3d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        # RMSNorm will normalize over the channel dimension; because RMSNorm normalizes over the last dimension,
        # we will permute channels to the last position before applying it.
        self.rmsnorm = nn.RMSNorm(out_channels, eps=1e-6, elementwise_affine=True)

        # Lazy BatchNorm2d will be initialized on first forward pass; expects (N, C, H, W)
        self.lazy_bn2d = nn.LazyBatchNorm2d()

        # Final classification head
        self.fc = nn.Linear(out_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
          1. ReplicationPad3d
          2. Conv3d -> GELU
          3. 1x1 Conv3d to project to out_channels
          4. RMSNorm over channels (permute channels to last dim for the op)
          5. Collapse depth dimension by averaging -> (B, C, H, W)
          6. LazyBatchNorm2d -> GELU
          7. Global average pooling over H,W -> (B, C)
          8. Linear classifier -> (B, num_classes)

        Args:
            x: Input tensor of shape (batch, in_channels, depth, height, width)

        Returns:
            logits: Tensor of shape (batch, num_classes)
        """
        # 1) Replication padding in 3D
        x = self.pad3d(x)

        # 2) First conv + activation
        x = self.conv3d_1(x)
        x = self.act(x)

        # 3) 1x1 conv to change channels
        x = self.conv3d_2(x)  # shape: (B, C_out, D, H, W)

        # 4) RMSNorm over channel dimension
        # Move channels to last dim: (B, D, H, W, C)
        x_perm = x.permute(0, 2, 3, 4, 1)
        x_norm = self.rmsnorm(x_perm)
        # Permute back to (B, C, D, H, W)
        x = x_norm.permute(0, 4, 1, 2, 3)

        # 5) Collapse depth dimension by averaging -> (B, C, H, W)
        x_2d = x.mean(dim=2)  # average over depth

        # 6) Lazy batch norm (initializes on first call) and activation
        x_2d = self.lazy_bn2d(x_2d)
        x_2d = self.act(x_2d)

        # 7) Global spatial average pooling over H and W -> (B, C)
        pooled = x_2d.mean(dim=(2, 3))

        # 8) Classification head
        logits = self.fc(pooled)

        return logits

# Configuration / default sizes
batch_size = 8
in_channels = 16
mid_channels = 32
out_channels = 64
depth = 8
height = 32
width = 32
kernel_size = 3
dilation = 2
# For kernel_size=3 and dilation=2 the effective kernel size is 5 -> padding of 2 on each side keeps spatial dims.
pad_sizes = (2, 2, 2, 2, 2, 2)  # (left, right, top, bottom, front, back)
num_classes = 10

def get_inputs() -> List[torch.Tensor]:
    """
    Returns a list containing a single input tensor suitable for the model:
      shape = (batch_size, in_channels, depth, height, width)
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs() -> List:
    """
    Returns the initialization parameters that the Model constructor expects,
    in the order:
      [in_channels, mid_channels, out_channels, kernel_size, pad_sizes, dilation, num_classes]
    """
    return [in_channels, mid_channels, out_channels, kernel_size, pad_sizes, dilation, num_classes]