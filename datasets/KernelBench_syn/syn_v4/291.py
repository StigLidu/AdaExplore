import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    3D feature extractor that demonstrates a combination of ReflectionPad3d,
    Conv3d, BatchNorm3d, ReLU6 and MaxPool3d, followed by simple global
    pooling and a small classifier head.

    Computation graph (high-level):
      x -> ReflectionPad3d -> Conv3d -> BatchNorm3d -> ReLU6 -> MaxPool3d -> [Global AvgPool, Global MaxReduce]
      -> concatenate -> Linear classifier -> L2 normalize logits

    This design shows multi-stage 3D processing, spatial reduction, two different
    global aggregation strategies and a final projection, combining the specified
    layers in a coherent pipeline.
    """
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        num_classes: int,
        kernel_size: int = 3,
        pool_kernel: int = 2,
        pad: int = 1
    ):
        """
        Initializes the model components.

        Args:
            in_channels (int): Number of input channels for the 3D volume.
            mid_channels (int): Number of channels produced by the first Conv3d.
            num_classes (int): Number of output classes for the classifier head.
            kernel_size (int, optional): Kernel size for Conv3d. Defaults to 3.
            pool_kernel (int, optional): Kernel/stride for MaxPool3d. Defaults to 2.
            pad (int, optional): Padding size for ReflectionPad3d. Defaults to 1.
        """
        super(Model, self).__init__()
        # Reflection pad to preserve spatial dims before convolution
        self.pad = nn.ReflectionPad3d(pad)

        # Simple 3D convolution block
        self.conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,  # padding is handled by ReflectionPad3d
            bias=False
        )
        self.bn1 = nn.BatchNorm3d(num_features=mid_channels)
        self.relu6 = nn.ReLU6(inplace=True)

        # Spatial downsampling
        self.pool = nn.MaxPool3d(kernel_size=pool_kernel, stride=pool_kernel)

        # Classifier head: takes concatenated [global_avg, global_max] of channel vectors
        self.fc = nn.Linear(in_features=mid_channels * 2, out_features=num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the 3D feature extractor and classifier.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_channels, D, H, W)

        Returns:
            torch.Tensor: L2-normalized logits with shape (batch_size, num_classes)
        """
        # Pad to keep consistent receptive field without zero-padding artifacts
        x = self.pad(x)                     # ReflectionPad3d

        # Convolutional feature extraction
        x = self.conv1(x)                   # Conv3d
        x = self.bn1(x)                     # BatchNorm3d
        x = self.relu6(x)                   # ReLU6

        # Reduce spatial resolution
        x = self.pool(x)                    # MaxPool3d

        # Two global aggregation strategies across spatial dims:
        # 1) Global average pooling to (batch, channels)
        batch = x.size(0)
        ga = F.adaptive_avg_pool3d(x, output_size=(1, 1, 1)).view(batch, -1)

        # 2) Global max reduction across spatial dims
        gm = x.view(batch, x.size(1), -1).amax(dim=2)

        # Concatenate aggregated descriptors and project to classes
        features = torch.cat([ga, gm], dim=1)  # shape: (batch, 2*channels)
        logits = self.fc(features)             # Linear projection to num_classes

        # L2-normalize the final outputs (stable)
        norm = logits.norm(p=2, dim=1, keepdim=True).clamp_min(1e-6)
        out = logits / norm
        return out

# Configuration / input shape parameters
batch_size = 8
in_channels = 3
mid_channels = 32
num_classes = 10
depth = 16
height = 32
width = 32
kernel_size = 3
pool_kernel = 2
pad = 1

def get_inputs():
    """
    Returns a list containing a single input tensor suitable for the Model.forward.
    Shape: (batch_size, in_channels, depth, height, width)
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization arguments for the Model constructor in the same order:
    [in_channels, mid_channels, num_classes, kernel_size, pool_kernel, pad]
    """
    return [in_channels, mid_channels, num_classes, kernel_size, pool_kernel, pad]