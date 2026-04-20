import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex 1D feature extractor that demonstrates a combination of circular padding,
    lazy convolution (in_channels inferred at first forward), batch normalization,
    channel-wise gating (squeeze-like non-learnable attention), conversion to 3D
    tensor followed by zero-padding in 3D and global spatial aggregation.

    The model returns a compact feature vector per batch item.
    """
    def __init__(
        self,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 2,
        circ_pad: int = 2,
        zero_pad: tuple = (0, 1, 0, 1, 0, 0),
    ):
        """
        Args:
            out_channels (int): Number of output channels for the conv layer.
            kernel_size (int, optional): Kernel size for convolution. Defaults to 5.
            stride (int, optional): Stride for convolution. Defaults to 2.
            circ_pad (int, optional): Circular padding value applied before conv. Defaults to 2.
            zero_pad (tuple, optional): ZeroPad3d padding (left,right, top,bottom, front,back).
                                        Defaults to (0,1,0,1,0,0).
        """
        super(Model, self).__init__()
        # Circular padding applied along the temporal (last) dimension
        self.circ_pad = nn.CircularPad1d(circ_pad)

        # LazyConv1d will infer in_channels at the first forward pass
        self.conv = nn.LazyConv1d(out_channels=out_channels, kernel_size=kernel_size, stride=stride)

        # BatchNorm for the conv output channels (will be initialized after first forward if needed)
        self.bn = nn.BatchNorm1d(out_channels)

        # Activation
        self.relu = nn.ReLU(inplace=True)

        # 3D zero padding - expects a 5D tensor (N, C, D, H, W)
        self.zero_pad3d = nn.ZeroPad3d(zero_pad)

        # Global spatial aggregation after padding -> produce compact per-channel features
        self.global_pool3d = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        1. Circular pad along time dimension.
        2. 1D convolution (lazy-initialized if necessary).
        3. Batch normalization and ReLU.
        4. Channel-wise gating using a simple squeeze (mean over time) + sigmoid.
        5. Reshape to 5D (N, C, D=1, H=time, W=1) and apply ZeroPad3d.
        6. Adaptive average pool to (1,1,1) spatially and return flattened feature vector.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, seq_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels) - compact features.
        """
        # 1. Circular padding (adds values at both ends of the temporal dimension)
        x = self.circ_pad(x)  # shape: (N, C, L + 2*circ_pad)

        # 2. Convolution -> BN -> ReLU
        x = self.conv(x)     # shape: (N, outC, L_out)
        x = self.bn(x)
        x = self.relu(x)

        # 3. Channel-wise gating (non-learnable squeeze + sigmoid)
        #    Squeeze: mean over temporal dimension -> (N, C, 1)
        se = x.mean(dim=2, keepdim=True)
        se = torch.sigmoid(se)
        x = x * se  # broadcast scaling across temporal positions

        # 4. Convert to 5D for ZeroPad3d: (N, C, D=1, H=L_out, W=1)
        x = x.unsqueeze(2).unsqueeze(4)

        # 5. Zero-pad in 3D space
        x = self.zero_pad3d(x)

        # 6. Global aggregation to (1,1,1) and flatten to (N, C)
        x = self.global_pool3d(x)  # (N, C, 1, 1, 1)
        x = x.view(x.size(0), -1)  # (N, C)

        return x

# Configuration variables
batch_size = 8
in_channels = 3
seq_length = 128

out_channels = 64
kernel_size = 5
stride = 2
circ_pad = 2
zero_pad = (0, 1, 0, 1, 0, 0)

def get_inputs():
    """
    Returns a list with a single input tensor matching expected model input:
    (batch_size, in_channels, seq_length)
    """
    x = torch.randn(batch_size, in_channels, seq_length)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for the Model constructor in order:
    out_channels, kernel_size, stride, circ_pad, zero_pad
    """
    return [out_channels, kernel_size, stride, circ_pad, zero_pad]