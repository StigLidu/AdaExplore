import math
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A 3D -> 2D hybrid model that:
      - Applies a 3D convolution over (D, H, W)
      - Uses LazyBatchNorm3d (lazy initialization of channels)
      - Activates with SiLU
      - Collapses the temporal/depth axis to apply an LPPool2d over spatial dimensions
      - Aggregates over depth and projects to a final class/logit vector

    The design shows a combination of 3D processing and 2D pooling, using nn.LazyBatchNorm3d
    and nn.LPPool2d from the provided layer list.
    """
    def __init__(self, num_classes: int = 64):
        """
        Args:
            num_classes (int): Number of output classes/features for the final linear layer.
        """
        super(Model, self).__init__()
        # 3D feature extractor
        self.conv3d = nn.Conv3d(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, kernel_size=3, padding=1, bias=False)
        # Lazy BatchNorm3d - will infer num_features from the first forward pass
        self.bn3d = nn.LazyBatchNorm3d()
        self.act = nn.SiLU()

        # 2D Lp pooling applied after collapsing the depth axis into the batch axis
        # norm_type=2 (L2 pool), kernel_size and stride set from module-level configuration
        self.lppool2d = nn.LPPool2d(norm_type=2, kernel_size=POOL_KERNEL, stride=POOL_STRIDE)

        # Final classifier/projection
        self.classifier = nn.Linear(OUT_CHANNELS * POOL_H_OUT * POOL_W_OUT, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
          x: (B, C_in, D, H, W)
        Returns:
          logits: (B, num_classes)
        """
        B = x.size(0)
        # 3D conv -> BN -> activation
        x = self.conv3d(x)               # (B, OUT_CHANNELS, D, H, W)
        x = self.bn3d(x)                 # lazy initialized on first pass
        x = self.act(x)

        # Move depth into the batch dimension so we can apply 2D pooling across spatial dims
        # Current shape: (B, OUT_CHANNELS, D, H, W)
        B, C, D, H, W = x.shape
        # Permute to (B, D, C, H, W) then collapse to (B*D, C, H, W)
        x = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)

        # Apply LPPool2d on 2D spatial dimensions
        x = self.lppool2d(x)             # (B*D, C, H_out, W_out)

        # Restore depth dimension: (B, D, C, H_out, W_out)
        x = x.view(B, D, C, POOL_H_OUT, POOL_W_OUT)

        # Aggregate over depth (temporal mean)
        x = x.mean(dim=1)                # (B, C, H_out, W_out)

        # Flatten spatial + channel dims and apply final classifier
        x = x.view(B, -1)                # (B, C * H_out * W_out)
        logits = self.classifier(x)      # (B, num_classes)
        return logits

# --------------------------
# Module-level configuration
# --------------------------
# Input tensor dimensions
BATCH_SIZE = 8
IN_CHANNELS = 3
DEPTH = 7
HEIGHT = 64
WIDTH = 48

# Convolution / feature map configuration
OUT_CHANNELS = 16  # number of output channels from Conv3d

# LPPool2d configuration
POOL_KERNEL = 3
POOL_STRIDE = 2

# Compute resulting spatial size after LPPool2d
# (using floor division consistent with standard pooling output calculation)
POOL_H_OUT = (HEIGHT - POOL_KERNEL) // POOL_STRIDE + 1
POOL_W_OUT = (WIDTH - POOL_KERNEL) // POOL_STRIDE + 1

# Classifier output
NUM_CLASSES = 64

def get_inputs():
    """
    Returns typical example inputs for the model:
      - A 5D tensor with shape (BATCH_SIZE, IN_CHANNELS, DEPTH, HEIGHT, WIDTH)
    """
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, DEPTH, HEIGHT, WIDTH)
    return [x]

def get_init_inputs():
    """
    Returns the list of initialization parameters for Model.__init__.
    Here we return the number of output classes used to initialize the classifier.
    """
    return [NUM_CLASSES]