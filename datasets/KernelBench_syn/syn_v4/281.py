import torch
import torch.nn as nn

# Configuration variables
batch_size = 8
channels = 64
depth = 16
height = 32
width = 32

# Pooling/shape hyperparameters
avgpool_kernel = (2, 2, 4)     # kernel for AvgPool3d -> reduces (D,H,W)
adaptive_out_len = 128         # output length for AdaptiveMaxPool1d

class Model(nn.Module):
    """
    Complex 3D feature aggregation model combining:
      - 3D average pooling to reduce spatial resolution,
      - lazy InstanceNorm3d to normalize channel activations,
      - adaptive 1D max pooling across flattened spatial dimension,
      - computation of channel-wise similarity matrices.

    Forward pipeline:
      x (B, C, D, H, W)
        -> AvgPool3d -> (B, C, D', H', W')
        -> LazyInstanceNorm3d -> same shape
        -> flatten spatial dims -> (B, C, L)
        -> AdaptiveMaxPool1d -> (B, C, P)
        -> center per-channel, compute channel similarity S (B, C, C)
        -> zero diagonal, sigmoid nonlinearity
    """
    def __init__(self,
                 avg_kernel=(2, 2, 4),
                 adaptive_output=128):
        super(Model, self).__init__()
        self.avgpool = nn.AvgPool3d(kernel_size=avg_kernel, stride=avg_kernel, ceil_mode=False)
        # LazyInstanceNorm3d will infer num_features at first forward based on input tensor
        self.inst_norm = nn.LazyInstanceNorm3d()
        # AdaptiveMaxPool1d maps variable-length flattened spatial dims to a fixed length
        self.adaptive_maxpool = nn.AdaptiveMaxPool1d(output_size=adaptive_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, D, H, W)
        Returns:
            Tensor of shape (B, C, C) representing a bounded (sigmoid) channel similarity matrix
            with zeroed diagonal for each sample in the batch.
        """
        # 1) Spatial reduction
        x = self.avgpool(x)  # (B, C, D', H', W')

        # 2) Channel-wise instance normalization (lazy-initialized)
        x = self.inst_norm(x)  # (B, C, D', H', W')

        # 3) Flatten spatial dimensions into a single sequence dimension L
        B, C, Dp, Hp, Wp = x.shape
        x = x.view(B, C, Dp * Hp * Wp)  # (B, C, L)

        # 4) Adaptive max pooling along the flattened spatial axis to fixed length P
        x = self.adaptive_maxpool(x)  # (B, C, P), where P = adaptive_output

        # 5) Center each channel's pooled features by subtracting the per-channel mean
        mean = x.mean(dim=2, keepdim=True)  # (B, C, 1)
        x_centered = x - mean  # (B, C, P)

        # 6) Channel-wise similarity via batched matrix multiplication
        #    S_ij = (x_i · x_j) / P  -> shape (B, C, C)
        P = x_centered.shape[2]
        S = torch.matmul(x_centered, x_centered.transpose(1, 2)) / float(max(P, 1))

        # 7) Zero out the diagonal (self-similarity) and apply sigmoid non-linearity
        eye = torch.eye(C, device=S.device, dtype=S.dtype).unsqueeze(0)  # (1, C, C)
        S = S * (1.0 - eye)  # zero diagonal per sample
        S = torch.sigmoid(S)  # bound similarities to (0,1)

        return S

def get_inputs():
    """
    Generates a batch of random 3D feature maps for testing.

    Returns:
        list: single-element list with input tensor of shape (batch_size, channels, depth, height, width)
    """
    x = torch.randn(batch_size, channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization inputs for the model. None required because modules are constructed with defaults.

    Returns:
        list: empty list
    """
    return []