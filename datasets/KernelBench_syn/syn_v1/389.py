import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Composite module that demonstrates interaction between 2D padding, a converted pseudo-3D tensor,
    AdaptiveAvgPool3d, and final 2D padding + simple normalization.
    
    Pipeline:
    - ReplicationPad2d applied to 2D input (batch, C, H, W)
    - Unsqueeze depth dim and repeat to create a pseudo-3D tensor (batch, C, D, H', W')
    - AdaptiveAvgPool3d to reduce (D, H', W') -> (out_d, out_h, out_w)
    - Collapse the pooled depth into channels -> (batch, C * out_d, out_h, out_w)
    - Apply ZeroPad2d to the 2D tensor
    - Subtract global per-sample mean and apply ReLU non-linearity
    """
    def __init__(self, pad_replication: int = 1, pad_zero: int = 1, adaptive_output_size=(2, 8, 8), depth_repeat: int = 4):
        super(Model, self).__init__()
        # Use replication padding to preserve boundary patterns before creating the pseudo-3D stack
        self.rep_pad = nn.ReplicationPad2d(pad_replication)
        # Adaptive 3D average pooling will reduce (D, H, W) to the specified output size
        self.adapt_pool3d = nn.AdaptiveAvgPool3d(adaptive_output_size)
        # Zero padding applied after collapsing depth into channels
        self.zero_pad = nn.ZeroPad2d(pad_zero)
        self.adaptive_output_size = tuple(adaptive_output_size)
        self.depth_repeat = int(depth_repeat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input of shape (N, C, H, W)
        Returns:
            torch.Tensor: Processed tensor (N, C * out_d, out_h + pad, out_w + pad) after pooling and padding
        """
        # 1) Replication pad in 2D
        x_padded = self.rep_pad(x)  # (N, C, H + 2*pad, W + 2*pad)

        # 2) Create a pseudo-3D tensor by adding a depth dimension and repeating
        x_3d = x_padded.unsqueeze(2)  # (N, C, 1, H', W')
        x_3d = x_3d.repeat(1, 1, self.depth_repeat, 1, 1)  # (N, C, D, H', W')

        # 3) Adaptive average pool in 3D to a small fixed spatial/temporal footprint
        pooled = self.adapt_pool3d(x_3d)  # (N, C, out_d, out_h, out_w)

        # 4) Collapse the pooled depth dimension into the channel dimension to fuse temporal info
        N, C, out_d, out_h, out_w = pooled.shape
        collapsed = pooled.view(N, C * out_d, out_h, out_w)  # (N, C * out_d, out_h, out_w)

        # 5) Apply zero padding in 2D space to prepare for final operations
        padded = self.zero_pad(collapsed)  # (N, C * out_d, out_h + 2*pad, out_w + 2*pad)

        # 6) Normalize per-sample mean (channel+spatial) and apply non-linearity
        mean_per_sample = padded.mean(dim=(1, 2, 3), keepdim=True)
        out = torch.relu(padded - mean_per_sample)

        return out

# Configuration variables for generating inputs and initialization
batch_size = 8
channels = 16
height = 64
width = 48
pad_replication = 2       # ReplicationPad2d padding (symmetric)
pad_zero = 1              # ZeroPad2d padding (symmetric)
adaptive_output_size = (4, 8, 6)  # (out_depth, out_height, out_width) for AdaptiveAvgPool3d
depth_repeat = 6

def get_inputs():
    """
    Returns example input tensors for the forward method.
    """
    x = torch.randn(batch_size, channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns the initialization arguments for the Model constructor in the order:
    (pad_replication, pad_zero, adaptive_output_size, depth_repeat)
    """
    return [pad_replication, pad_zero, adaptive_output_size, depth_repeat]