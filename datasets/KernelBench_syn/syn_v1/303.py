import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class Model(nn.Module):
    """
    Volumetric feature extractor that combines 3D convolution, Lp pooling,
    channel-wise Dropout1d and an Adaptive Log-Softmax head for large-vocabulary classification.
    
    The forward pass:
      - 3D convolution to mix spatial information and channels
      - Lp pooling (nn.LPPool3d) to perform a power-average downsampling
      - ReLU activation
      - Reshape to (batch, channels, length) and apply Dropout1d to zero entire channels
      - Aggregate over the spatial length with mean to obtain a per-channel descriptor
      - Linear projection to the adaptive-softmax input dimensionality
      - AdaptiveLogSoftmaxWithLoss to compute loss against provided targets
    """
    def __init__(
        self,
        in_channels: int,
        pool_kernel: int,
        lp_norm: int,
        mid_channels: int,
        in_features: int,
        num_classes: int,
        cutoffs: Optional[List[int]],
        dropout: float = 0.5,
    ):
        """
        Initializes the module.

        Args:
            in_channels (int): Number of input channels in the volumetric data.
            pool_kernel (int): Kernel size for LPPool3d (also used as stride).
            lp_norm (int): The 'p' norm for LPPool3d (e.g., 1, 2, ...).
            mid_channels (int): Number of channels produced by the initial Conv3d.
            in_features (int): Feature dimensionality expected by AdaptiveLogSoftmaxWithLoss.
            num_classes (int): Number of target classes for classification.
            cutoffs (List[int] or None): Cutoff boundaries for adaptive softmax.
            dropout (float): Dropout probability for Dropout1d.
        """
        super(Model, self).__init__()
        # 3D convolution to extract local volumetric features
        self.conv3d = nn.Conv3d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, padding=1)
        # Lp pooling to reduce spatial dimensions with a power-average
        self.lppool = nn.LPPool3d(norm_type=lp_norm, kernel_size=pool_kernel, stride=pool_kernel)
        # Channel-wise dropout for 1D inputs (we'll reshape the pooled tensor to (B, C, L))
        self.dropout1d = nn.Dropout1d(p=dropout)
        # Linear projection from pooled channel descriptors to the adaptive-softmax input dim
        self.project = nn.Linear(mid_channels, in_features)
        # Adaptive softmax for efficient handling of many classes
        self.adaptive_logsoftmax = nn.AdaptiveLogSoftmaxWithLoss(in_features=in_features,
                                                                 n_classes=num_classes,
                                                                 cutoffs=cutoffs)
        # Small attribute bookkeeping
        self.mid_channels = mid_channels
        self.in_features = in_features

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing the adaptive-softmax loss for the inputs.

        Args:
            x (torch.Tensor): Input volumetric tensor of shape (B, C_in, D, H, W).
            target (torch.Tensor): LongTensor of shape (B,) with class indices in [0, num_classes-1].

        Returns:
            torch.Tensor: Scalar loss tensor produced by AdaptiveLogSoftmaxWithLoss.
        """
        # conv -> LpPool -> relu
        x = self.conv3d(x)
        x = self.lppool(x)
        x = F.relu(x)

        # x is now (B, C, D', H', W'). Reshape to (B, C, L) for Dropout1d which zeros entire channels.
        B, C, Dp, Hp, Wp = x.shape
        x = x.view(B, C, -1)  # (B, C, L)

        # Channel-wise dropout
        x = self.dropout1d(x)  # still (B, C, L)

        # Aggregate spatially to get a per-channel descriptor: (B, C)
        x = x.mean(dim=2)

        # Project to adaptive-softmax input dimensionality
        x = self.project(x)  # (B, in_features)

        # AdaptiveLogSoftmaxWithLoss expects (input, target) and returns a loss object/tuple.
        out = self.adaptive_logsoftmax(x, target)

        # Different PyTorch versions may return a namedtuple with .loss or a tuple (loss, output).
        if isinstance(out, tuple) or isinstance(out, list):
            loss = out[0]
        else:
            # Named tuple-like object (has .loss)
            try:
                loss = out.loss
            except Exception:
                # Fallback: try index access
                loss = out[0]

        return loss


# Module-level configuration variables (these are used by get_inputs and get_init_inputs).
batch_size = 8
in_channels = 3
depth = 32
height = 64
width = 64

pool_kernel = 2
lp_norm = 2  # p=2 (Euclidean-like pooling)
mid_channels = 64
in_features = 128

num_classes = 1000
cutoffs = [200, 600]  # must be monotonic and less than num_classes
dropout = 0.4


def get_inputs():
    """
    Returns runtime input tensors: the volumetric input and corresponding target labels.
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    target = torch.randint(low=0, high=num_classes, size=(batch_size,), dtype=torch.long)
    return [x, target]


def get_init_inputs():
    """
    Returns the constructor initialization parameters in the same order as Model.__init__.
    """
    return [in_channels, pool_kernel, lp_norm, mid_channels, in_features, num_classes, cutoffs, dropout]