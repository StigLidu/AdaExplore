import torch
import torch.nn as nn

# Configuration
batch_size = 8
channels = 16
depth = 32
height = 8
width = 8
proj_dim = 64  # output projection dimension

class Model(nn.Module):
    """
    A moderately complex 3D-processing module which:
    - Applies BatchNorm3d over the input volume
    - Applies a CELU non-linearity
    - Rearranges and collapses spatial dimensions to compute per-depth statistics
    - Projects the per-depth summary with a learnable projection matrix
    - Applies a HardTanh clamp to the final projected outputs

    Input shape: (N, C, D, H, W)
    Output shape: (N, proj_dim)
    """
    def __init__(self, in_channels: int = channels, d: int = depth, out_dim: int = proj_dim):
        super(Model, self).__init__()
        # Batch normalization across channels for 5D input
        self.bn = nn.BatchNorm3d(in_channels)
        # Non-linearities
        self.celu = nn.CELU()
        # Final clamp
        self.hardtanh = nn.Hardtanh(min_val=-3.0, max_val=3.0)
        # Learnable projection from per-depth summary (D) -> out_dim
        # We register as Parameter to keep the weight matrix trainable
        self.register_parameter("proj_weight", nn.Parameter(torch.randn(d, out_dim)))
        # Optional bias for projection
        self.register_parameter("proj_bias", nn.Parameter(torch.zeros(out_dim)))

        # Save shapes for use in forward (helps readability)
        self._depth = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        1. BatchNorm3d over (N, C, D, H, W)
        2. CELU activation
        3. Permute to (N, D, C, H, W) and collapse (C*H*W) so each depth slice is summarized
        4. Compute mean across the collapsed spatial axis -> (N, D)
        5. Linear projection using proj_weight -> (N, out_dim)
        6. Add bias and apply HardTanh clamp

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, D, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (N, out_dim)
        """
        # 1. Normalize channels across the 5D input
        y = self.bn(x)

        # 2. Non-linearity
        y = self.celu(y)

        # 3. Move depth to second dim explicitly and collapse the rest of spatial dims per-depth
        #    y shape currently: (N, C, D, H, W)
        #    after permute: (N, D, C, H, W)
        y = y.permute(0, 2, 1, 3, 4)

        # 4. Collapse channel and spatial dims into a single axis for each depth slice
        N, D, C, H, W = y.shape
        # ensure expected depth matches init
        if D != self._depth:
            # Fallback to dynamic projection trimming/padding if different depth encountered
            # Here we will adapt the projection weight by slicing or repeating as needed
            if D < self._depth:
                proj_w = self.proj_weight[:D, :]
            else:
                # tile/truncate to match larger D
                repeats = (D + self._depth - 1) // self._depth
                proj_w = self.proj_weight.repeat(repeats, 1)[:D, :]
        else:
            proj_w = self.proj_weight

        y = y.reshape(N, D, C * H * W)

        # 5. Per-depth summary: mean over the collapsed spatial axis -> (N, D)
        y = y.mean(dim=2)

        # 6. Linear projection (batch matmul)
        # y: (N, D), proj_w: (D, out_dim) -> out: (N, out_dim)
        y = torch.matmul(y, proj_w) + self.proj_bias

        # 7. Final clamp for numerical stability and non-linearity
        y = self.hardtanh(y)

        return y

def get_inputs():
    """
    Create a random 5D tensor suitable for BatchNorm3d:
    Shape: (batch_size, channels, depth, height, width)
    """
    x = torch.randn(batch_size, channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Return initialization parameters if any are needed externally.
    Here, model parameters are registered within the module itself, so no external init inputs are required.
    """
    return []