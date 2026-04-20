import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex module that extracts local patches with Unfold, computes
    patch-wise statistics to form attention weights, combines these with
    AvgPool2d summaries, applies a lazy BatchNorm2d, non-linearity,
    and upsamples back to the input resolution.

    Computation steps:
    1. avg = AvgPool2d(x) to produce a coarse spatial summary.
    2. patches = Unfold(x) to extract flattened sliding patches.
    3. Compute per-patch mean across the patch area and derive attention
       weights with softmax over spatial locations.
    4. Use the attention weights to compute a weighted summary of the
       pooled map, broadcast it back to the pooled spatial grid and add
       to the pooled map.
    5. Normalize with LazyBatchNorm2d, apply ReLU and upsample to the
       original resolution.
    """
    def __init__(self, kernel_size: int = 4, stride: int = 4, pool_kernel: int = 4):
        """
        Args:
            kernel_size (int): Kernel size used by Unfold for patch extraction.
            stride (int): Stride used by Unfold and as upsample factor at the end.
            pool_kernel (int): Kernel size used by AvgPool2d to create coarse summaries.
        """
        super(Model, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pool_kernel = pool_kernel

        # Unfold to extract sliding local blocks
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, stride=self.stride)

        # Average pooling to create a coarse spatial map (should align with unfold grid)
        self.avgpool = nn.AvgPool2d(kernel_size=self.pool_kernel, stride=self.pool_kernel)

        # Lazy batch norm so the channel dimension can be inferred on first forward
        self.bn = nn.LazyBatchNorm2d()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Tensor of shape (B, C, H, W) -- enhanced & normalized version of input.
        """
        B, C, H, W = x.shape

        # 1) Coarse spatial summary
        avg = self.avgpool(x)                       # (B, C, H_out, W_out)
        H_out, W_out = avg.shape[2], avg.shape[3]
        L = H_out * W_out                           # number of pooled spatial locations

        # 2) Extract patches
        patches = self.unfold(x)                    # (B, C * k*k, L)
        # Reshape to separate channel and local patch elements: (B, C, k*k, L)
        k2 = self.kernel_size * self.kernel_size
        patches_reshaped = patches.view(B, C, k2, L)

        # 3) Compute patch-wise statistics -> per-patch mean per channel
        patch_mean = patches_reshaped.mean(dim=2)   # (B, C, L)
        # Aggregate across channels to get a single scalar per patch location
        spatial_scores = patch_mean.mean(dim=1)     # (B, L)

        # Softmax over spatial locations to get attention weights for each patch
        weights = F.softmax(spatial_scores, dim=1)  # (B, L)

        # 4) Use weights to compute a weighted summary of the pooled map
        avg_flat = avg.view(B, C, L)                # (B, C, L)
        # Weighted sum across spatial locations -> (B, C)
        weighted_sum = (avg_flat * weights.unsqueeze(1)).sum(dim=2)

        # Broadcast weighted summary back to pooled spatial grid and add residual
        weighted_map = weighted_sum.unsqueeze(-1).expand(B, C, L).contiguous()
        weighted_map = weighted_map.view(B, C, H_out, W_out)  # (B, C, H_out, W_out)

        # Combine coarse map with weighted map
        out = avg + weighted_map

        # 5) Normalize, non-linearity and upsample to original resolution
        out = self.bn(out)          # LazyBatchNorm2d will infer C on first forward
        out = F.relu(out)
        # Upsample back to original input resolution (use stride as upsample factor)
        out = F.interpolate(out, scale_factor=self.stride, mode='nearest')

        # Ensure output spatial dims match input (in case of rounding)
        if out.shape[2] != H or out.shape[3] != W:
            out = F.interpolate(out, size=(H, W), mode='nearest')

        return out

# Configuration / sizes
batch_size = 8
channels = 16
height = 32
width = 32

kernel_size = 4
stride = 4
pool_kernel = 4

def get_inputs():
    """
    Returns:
        List containing one input tensor shaped (batch_size, channels, height, width)
    """
    x = torch.randn(batch_size, channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters for the Model: kernel_size, stride, pool_kernel
    """
    return [kernel_size, stride, pool_kernel]