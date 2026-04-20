import torch
import torch.nn as nn

# Configuration: shapes and pooling parameters
batch_size = 8
channels = 16
depth = 8
height = 64
width = 64

# Pooling configuration for 3D and 2D pools
avg3d_kernel = (2, 2, 2)   # reduces (D, H, W)
avg3d_stride = (2, 2, 2)
avg3d_padding = 0

avg2d_kernel = 2
avg2d_stride = 2
avg2d_padding = 0

class Model(nn.Module):
    """
    Complex model combining LazyInstanceNorm3d, AvgPool3d and AvgPool2d.
    The forward path:
      1. Lazy instance normalization on the 5D input (N, C, D, H, W).
      2. ReLU activation.
      3. 3D average pooling to reduce spatial-temporal dimensions -> (N, C, D', H', W').
      4. Collapse the depth dimension into channels to obtain a 4D tensor -> (N, C * D', H', W').
      5. 2D average pooling on the collapsed tensor.
      6. Channel-wise gating computed from spatial averages (sigmoid of channel means).
      7. Reapply gating and reshape back to a 5D tensor (N, C, D', H'', W'') as final output.
    """
    def __init__(
        self,
        avg3d_kernel=avg3d_kernel,
        avg3d_stride=avg3d_stride,
        avg3d_padding=avg3d_padding,
        avg2d_kernel=avg2d_kernel,
        avg2d_stride=avg2d_stride,
        avg2d_padding=avg2d_padding,
    ):
        super(Model, self).__init__()
        # LazyInstanceNorm3d will infer num_features on first forward pass
        self.inst_norm3d = nn.LazyInstanceNorm3d()
        # Pooling layers as provided
        self.avg3d = nn.AvgPool3d(kernel_size=avg3d_kernel, stride=avg3d_stride, padding=avg3d_padding)
        self.avg2d = nn.AvgPool2d(kernel_size=avg2d_kernel, stride=avg2d_stride, padding=avg2d_padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (N, C, D, H, W)

        Returns:
            Tensor of shape (N, C, D', H_out, W_out) after normalization, pooling,
            channel-wise gating and reshaping.
        """
        # 1. Instance normalization (lazy)
        x_norm = self.inst_norm3d(x)

        # 2. Non-linear activation
        x_act = torch.relu(x_norm)

        # 3. 3D average pooling: reduces D, H, W -> (N, C, D1, H1, W1)
        p3 = self.avg3d(x_act)

        # 4. Collapse depth into channels to create a 4D tensor for 2D pooling:
        #    (N, C, D1, H1, W1) -> (N, C * D1, H1, W1)
        N, C, D1, H1, W1 = p3.shape
        p3_collapsed = p3.view(N, C * D1, H1, W1)

        # 5. 2D average pooling on collapsed tensor -> (N, C * D1, H2, W2)
        p2 = self.avg2d(p3_collapsed)

        # 6. Channel-wise gating: compute spatial mean per channel and apply sigmoid
        #    channel_summary shape: (N, C * D1)
        channel_summary = p2.mean(dim=(2, 3))
        gating = torch.sigmoid(channel_summary).unsqueeze(-1).unsqueeze(-1)  # (N, C*D1, 1, 1)

        # 7. Apply gating and reshape back to 5D: (N, C, D1, H2, W2)
        gated = p2 * gating
        H2, W2 = p2.shape[2], p2.shape[3]
        out = gated.view(N, C, D1, H2, W2)

        return out

def get_inputs():
    """
    Returns a list containing a single input tensor of shape (batch_size, channels, depth, height, width).
    """
    x = torch.randn(batch_size, channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Return initialization parameters used in the model constructor.
    This returns pooling kernel/stride/padding tuples for 3D and simple ints for 2D.
    """
    return [avg3d_kernel, avg3d_stride, avg3d_padding, avg2d_kernel, avg2d_stride, avg2d_padding]