import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex module that combines PixelUnshuffle, AvgPool1d, and LazyInstanceNorm3d.
    Pipeline:
      - PixelUnshuffle to trade spatial resolution for channels
      - Flatten spatial dims and apply AvgPool1d to reduce spatial length
      - Reshape into 5D tensor (N, C', D, H2, W2) and apply LazyInstanceNorm3d
      - Globally gate the normalized features using a sigmoid of the original input mean
      - Flatten spatial volume back to a sequence and apply a second AvgPool1d followed by ReLU

    This creates a computation pattern that mixes vision rearrangement, 1D pooling on
    flattened spatial sequences, and 3D instance normalization on a volumetric view.
    """
    def __init__(self, downscale: int = 2, pool_k: int = 4, depth: int = 2):
        """
        Args:
            downscale (int): PixelUnshuffle downscale factor (r). Assumes input H,W divisible by r.
            pool_k (int): Kernel size / stride for the AvgPool1d operations (reduces spatial length).
            depth (int): Depth dimension to reshape pooled sequence into for InstanceNorm3d.
        """
        super(Model, self).__init__()
        self.downscale = downscale
        self.pool_k = pool_k
        self.depth = depth

        # Layers from provided list
        self.pixel_unshuffle = nn.PixelUnshuffle(self.downscale)
        self.avgpool1 = nn.AvgPool1d(kernel_size=self.pool_k, stride=self.pool_k)
        # LazyInstanceNorm3d will infer num_features (channels) at the first forward call
        self.linstnorm3d = nn.LazyInstanceNorm3d()

        # Reuse the same pooling layer for the second reduction (behaviorally fine)
        self.avgpool2 = nn.AvgPool1d(kernel_size=self.pool_k, stride=self.pool_k)

        # Non-linearity
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input image tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Processed tensor of shape (N, C', L_final) where C' = C * downscale^2
                          and L_final is the final pooled sequence length.
        """
        # Validate input dimensions
        if x.dim() != 4:
            raise ValueError("Input x must be a 4D tensor (N, C, H, W)")

        N, C, H, W = x.shape

        # PixelUnshuffle: (N, C, H, W) -> (N, C * r^2, H/r, W/r)
        y = self.pixel_unshuffle(x)
        N, C_un, H_u, W_u = y.shape

        # Flatten spatial dims to a sequence for 1D pooling: (N, C_un, H_u*W_u)
        y_seq = y.view(N, C_un, H_u * W_u)

        # First AvgPool1d reduces spatial length
        if (H_u * W_u) % self.pool_k != 0:
            # For simplicity enforce divisibility to keep reshapes simple
            raise ValueError("After PixelUnshuffle, (H/r * W/r) must be divisible by pool_k")
        y_pooled = self.avgpool1(y_seq)  # (N, C_un, L_p)
        L_p = y_pooled.shape[2]

        # Reshape pooled sequence into a volumetric 5D tensor for InstanceNorm3d:
        # Choose H2 = H_u // 2 (must divide), compute W2 so that depth * H2 * W2 == L_p
        if H_u % 2 != 0:
            raise ValueError("H/r must be divisible by 2 for the chosen reshape strategy")
        H2 = H_u // 2
        if (L_p) % (self.depth * H2) != 0:
            raise ValueError("Cannot reshape pooled sequence into (depth, H2, W2) with current parameters")
        W2 = (L_p) // (self.depth * H2)

        # Final 5D shape: (N, C_un, depth, H2, W2)
        y_5d = y_pooled.view(N, C_un, self.depth, H2, W2)

        # Apply LazyInstanceNorm3d (num_features inferred on first call)
        y_norm = self.linstnorm3d(y_5d)

        # Compute a per-batch gating scalar from the original input and apply it
        # s shape: (N, 1, 1, 1, 1) to broadcast across channels and spatial dims
        s = torch.sigmoid(torch.mean(x, dim=[1, 2, 3], keepdim=True)).view(N, 1, 1, 1, 1)
        y_gated = y_norm * s

        # Flatten depth and spatial dims back into a sequence (N, C_un, L_p)
        y_flat = y_gated.view(N, C_un, -1)

        # Second AvgPool1d to further reduce sequence length
        # Ensure divisibility for pooling; if not divisible, AvgPool1d will floor; we prefer divisible for determinism
        if y_flat.shape[2] % self.pool_k != 0:
            # If not divisible, pad the sequence at the end to make it divisible
            pad_len = self.pool_k - (y_flat.shape[2] % self.pool_k)
            pad = y_flat.new_zeros((N, C_un, pad_len))
            y_flat = torch.cat([y_flat, pad], dim=2)
        y_final_seq = self.avgpool2(y_flat)  # (N, C_un, L_final)

        # Final non-linearity
        out = self.relu(y_final_seq)

        return out

# Module-level configuration (matches assumptions in the code)
batch_size = 8
in_channels = 3
height = 64
width = 64

# These initialization parameters must satisfy the divisibility constraints used in the model.
downscale = 2  # pixel unshuffle factor r
pool_k = 4     # AvgPool1d kernel/stride (first pooling reduces H_u*W_u by factor pool_k)
depth = 2      # depth for reshaping into 5D (depth * H2 * W2 == L_p)

def get_inputs():
    """
    Returns:
        list: single input tensor [x] shaped (batch_size, in_channels, height, width)
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns:
        list: initialization parameters for Model: [downscale, pool_k, depth]
    """
    return [downscale, pool_k, depth]