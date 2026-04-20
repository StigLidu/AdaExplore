import torch
import torch.nn as nn
from typing import List, Any, Tuple

class Model(nn.Module):
    """
    Complex module combining LazyInstanceNorm3d, AvgPool2d and PReLU with
    a gating mechanism and depth-to-batch reshaping to apply 2D pooling
    across the spatial dimensions for each depth slice.

    Forward pass:
      1. Instance normalization over 3D feature maps (lazy num_features).
      2. Channel-wise gating computed from mean activations (sigmoid).
      3. Reshape to merge depth into batch dimension so AvgPool2d operates
         slice-wise over (H, W) for every (batch*depth, channel) pair.
      4. Apply AvgPool2d, then PReLU activation.
      5. Reshape back to original 5D layout.
    """
    def __init__(
        self,
        pool_kernel: Tuple[int, int],
        pool_stride: Tuple[int, int] = None,
        eps: float = 1e-5,
        prelu_num_parameters: int = 1,
        prelu_init: float = 0.25,
    ):
        """
        Args:
            pool_kernel (tuple): Kernel size for 2D average pooling (h, w).
            pool_stride (tuple, optional): Stride for 2D average pooling (h, w).
            eps (float, optional): Epsilon for InstanceNorm3d.
            prelu_num_parameters (int, optional): Number of PReLU parameters (1 or number of channels).
            prelu_init (float, optional): Initial value for PReLU parameter(s).
        """
        super(Model, self).__init__()
        # Lazy instance norm: num_features inferred at first forward pass
        self.instancenorm = nn.LazyInstanceNorm3d(eps=eps, affine=True)
        # 2D average pooling to be applied on per-depth slices
        self.avgpool = nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_stride)
        # Parametric ReLU, can be channel-wise if given num_parameters equals number of channels
        self.prelu = nn.PReLU(num_parameters=prelu_num_parameters, init=prelu_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, D, H, W).
        
        Returns:
            torch.Tensor: Output tensor of shape (B, C, D, H_out, W_out),
                          where H_out and W_out depend on pooling.
        """
        # 1) Normalize across channels for each sample and spatial location (3D instance norm)
        x_norm = self.instancenorm(x)  # (B, C, D, H, W)

        # 2) Compute a simple channel-wise gating signal from spatial & depth means
        #    shape: (B, C, 1, 1, 1)
        channel_summary = x_norm.mean(dim=[2, 3, 4], keepdim=True)
        gate = torch.sigmoid(channel_summary)
        x_gated = x_norm * gate  # gated features, broadcasting over spatial dims

        # 3) Merge depth into batch so 2D pooling works on each depth slice independently
        b, c, d, h, w = x_gated.shape
        # permute to (B, D, C, H, W) then reshape to (B*D, C, H, W)
        x_slices = x_gated.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)

        # 4) Apply 2D average pooling slice-wise, then PReLU activation
        pooled = self.avgpool(x_slices)  # (B*D, C, H_out, W_out)
        activated = self.prelu(pooled)

        # 5) Reshape back to (B, C, D, H_out, W_out)
        h_out, w_out = activated.shape[2], activated.shape[3]
        out = activated.reshape(b, d, c, h_out, w_out).permute(0, 2, 1, 3, 4)

        return out

# Configuration / default inputs
batch_size = 4
channels = 8
depth = 6
height = 32
width = 32

pool_kernel = (3, 3)
pool_stride = (2, 2)
eps = 1e-5
prelu_num_parameters = channels  # per-channel PReLU parameters
prelu_init = 0.2

def get_inputs() -> List[torch.Tensor]:
    """
    Creates a realistic 5D tensor input for the model: (B, C, D, H, W).
    """
    x = torch.randn(batch_size, channels, depth, height, width)
    return [x]

def get_init_inputs() -> List[Any]:
    """
    Returns initialization arguments for Model.__init__ in the same order.
    """
    return [pool_kernel, pool_stride, eps, prelu_num_parameters, prelu_init]