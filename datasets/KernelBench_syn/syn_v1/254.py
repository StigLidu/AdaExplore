import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    A composite model that demonstrates interplay between 3D pooling,
    2D upsampling, and lazy instance normalization. The model:
      1. Applies a 3D max pooling to reduce spatial + depth resolution.
      2. Collapses the pooled depth into the channel dimension to form a 4D tensor.
      3. Applies LazyInstanceNorm2d (lazy initialization based on channels).
      4. Activates with ReLU and upsamples spatially using bilinear interpolation.
      5. Restores a pseudo-5D layout (split channels back into channel and depth),
         collapses depth via averaging, and merges with the original depth-averaged input
         via a learned gating parameter.
    The model is intentionally designed to mix 3D and 2D operations and to exercise lazy
    initialization behavior.
    """
    def __init__(self,
                 pool_kernel=(2, 2, 2),
                 pool_stride=(2, 2, 2),
                 upsample_scale=2):
        """
        Initializes layers used in the computation.

        Args:
            pool_kernel (tuple): kernel size for MaxPool3d.
            pool_stride (tuple): stride for MaxPool3d.
            upsample_scale (int): spatial upsampling factor used by UpsamplingBilinear2d.
        """
        super(Model, self).__init__()
        # 3D max pooling to reduce depth and spatial dims
        self.pool3d = nn.MaxPool3d(kernel_size=pool_kernel, stride=pool_stride)
        # Bilinear upsampling for 2D feature maps
        self.upsample2d = nn.UpsamplingBilinear2d(scale_factor=upsample_scale)
        # Lazy instance norm will be initialized on first forward pass when channel dim is known
        self.inst_norm = nn.LazyInstanceNorm2d()
        # Learnable scalar used as a gating parameter; will be squashed by sigmoid in forward
        self.gate_param = nn.Parameter(torch.tensor(0.0))

        # Save configuration for forward checks / documentation
        self.pool_kernel = pool_kernel
        self.pool_stride = pool_stride
        self.upsample_scale = upsample_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation.

        Args:
            x (torch.Tensor): Input tensor with shape (N, C, D, H, W).

        Returns:
            torch.Tensor: Output tensor with shape (N, 2*C, H, W) formed by concatenating
                          the original depth-averaged features and a processed branch.
        """
        # Validate input dims
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input (N,C,D,H,W), got shape {tuple(x.shape)}")

        # Step 1: 3D max pooling -> reduces depth and spatial resolution
        pooled = self.pool3d(x)  # shape: (N, C, D_p, H_p, W_p)

        N, C, Dp, Hp, Wp = pooled.shape

        # Step 2: Collapse depth into channels to obtain a 4D tensor for 2D ops:
        # new_channels = C * Dp
        collapsed = pooled.view(N, C * Dp, Hp, Wp)  # shape: (N, C*Dp, Hp, Wp)

        # Step 3: Lazy instance normalization (initializes based on C*Dp on first forward)
        normalized = self.inst_norm(collapsed)

        # Step 4: Non-linearity
        activated = F.relu(normalized)

        # Step 5: Upsample spatially back towards original H/W (expects Hp * scale = H)
        upsampled = self.upsample2d(activated)  # shape: (N, C*Dp, Hp*scale, Wp*scale)

        # Sanity check: ensure spatial dims match the input's H and W
        expected_H = x.shape[3]
        expected_W = x.shape[4]
        if upsampled.shape[2] != expected_H or upsampled.shape[3] != expected_W:
            # If sizes do not match exactly, perform an explicit interpolation to match original dims
            upsampled = F.interpolate(upsampled, size=(expected_H, expected_W), mode='bilinear', align_corners=False)

        # Step 6: Restore a pseudo-5D layout by splitting the channel dimension back into (C, Dp)
        restored = upsampled.view(N, C, Dp, expected_H, expected_W)  # shape: (N, C, Dp, H, W)

        # Step 7: Collapse the depth axis by averaging to produce a 4D feature map
        branch_feat = restored.mean(dim=2)  # shape: (N, C, H, W)

        # Baseline: depth-averaged features from the original input
        input_depth_avg = x.mean(dim=2)  # shape: (N, C, H, W)

        # Step 8: Gated combination between original depth-averaged features and processed branch
        gate = torch.sigmoid(self.gate_param)  # scalar in (0,1)
        combined = gate * input_depth_avg + (1.0 - gate) * branch_feat  # shape: (N, C, H, W)

        # Step 9: Concatenate the baseline and combined features to form a richer representation
        out = torch.cat([input_depth_avg, combined], dim=1)  # shape: (N, 2*C, H, W)

        return out


# Configuration variables for input generation
batch_size = 4
channels = 3
depth = 8     # must be divisible by pool kernel depth (2) in default config
height = 32   # must be compatible with pooling+upsampling (e.g., 32 -> pool (16) -> upsample (32))
width = 32

def get_inputs():
    """
    Returns a list of inputs for the model's forward method.
    The returned input matches the expected 5D shape (N, C, D, H, W).
    """
    x = torch.randn(batch_size, channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization inputs required to construct the Model.
    There are none required in this example beyond default constructor args.
    """
    return []