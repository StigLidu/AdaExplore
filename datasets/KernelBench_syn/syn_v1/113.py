import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex 3D-processing model that demonstrates a pipeline combining:
    - Group Normalization over channels
    - Threshold activation for sparsity
    - Lazy BatchNorm3d (lazy-initialized) applied to depth-reduced tensors
    - Attention-like weighting across the depth dimension followed by a weighted sum

    The forward pass expects a 5D tensor (N, C, D, H, W) and returns a 2D tensor (N, C)
    representing a channel-wise aggregated descriptor for each batch element.
    """
    def __init__(self, channels: int, num_groups: int, threshold: float = 0.0, thr_value: float = 0.0, thr_inplace: bool = False):
        """
        Initializes the modules used in the pipeline.

        Args:
            channels (int): Number of channels in the input tensor.
            num_groups (int): Number of groups for GroupNorm (must divide channels).
            threshold (float): Threshold value for nn.Threshold (elements <= threshold are set to thr_value).
            thr_value (float): Value to replace elements below or equal to threshold.
            thr_inplace (bool): Whether the threshold operation should be inplace.
        """
        super(Model, self).__init__()

        # GroupNorm operates on the channel dimension; channels must be divisible by num_groups.
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=channels)

        # Threshold activation: elements <= threshold are set to thr_value.
        self.threshold = nn.Threshold(threshold, thr_value, inplace=thr_inplace)

        # LazyBatchNorm3d: will infer num_features at first forward call when given a (N, C, D, H, W) tensor.
        self.lazy_bn3d = nn.LazyBatchNorm3d()

        # Small learnable linear projection applied after depth aggregation (per-channel).
        # Implemented as a 1x1 conv equivalent using nn.Linear for shape (N, C).
        self.post_proj = nn.Linear(channels, channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation:
          1. Apply GroupNorm across channels: preserves shape (N, C, D, H, W).
          2. Apply Threshold non-linearity to introduce sparsity.
          3. Spatial average pool (H, W) to produce (N, C, D).
          4. Treat (N, C, D) as (N, C, D, 1, 1) and apply LazyBatchNorm3d (N, C, D, H, W format).
          5. Squeeze back to (N, C, D) and compute depth-wise softmax to obtain attention weights.
          6. Weighted sum across depth to produce (N, C).
          7. Final linear projection and a residual-skip + GELU nonlinearity for richer representation.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (N, C).
        """
        # 1. Group normalization
        y = self.groupnorm(x)

        # 2. Threshold activation
        y = self.threshold(y)

        # 3. Spatial average pooling over H and W -> shape (N, C, D)
        y_spatial = y.mean(dim=[3, 4])  # average over H and W

        # 4. Prepare for BatchNorm3d which expects (N, C, D, H, W). We use H=W=1 here.
        bn_input = y_spatial.unsqueeze(-1).unsqueeze(-1)  # (N, C, D, 1, 1)
        bn_output = self.lazy_bn3d(bn_input)  # Lazy init occurs here if needed

        # 5. Collapse spatial singleton dims -> (N, C, D)
        bn_squeezed = bn_output.view(bn_output.size(0), bn_output.size(1), bn_output.size(2))

        # 6. Compute attention weights across depth (softmax over D axis)
        attn = F.softmax(bn_squeezed, dim=2)

        # 7. Weighted sum across depth to produce (N, C)
        weighted = (y_spatial * attn).sum(dim=2)

        # 8. Final projection + residual + GELU
        proj = self.post_proj(weighted)  # (N, C)
        out = F.gelu(proj + weighted)    # residual connection for stability and expressiveness

        return out

# Configuration: choose sizes that are compatible with GroupNorm num_groups
batch_size = 8
channels = 32         # should be divisible by num_groups below
depth = 8
height = 16
width = 16

def get_inputs():
    """
    Returns a single input tensor matching the expected input shape for the Model:
    (batch_size, channels, depth, height, width)
    """
    x = torch.randn(batch_size, channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for Model:
    [channels, num_groups, threshold, thr_value, thr_inplace]

    - num_groups must divide channels.
    """
    num_groups = 8
    threshold = 0.0
    thr_value = 0.0
    thr_inplace = False
    return [channels, num_groups, threshold, thr_value, thr_inplace]