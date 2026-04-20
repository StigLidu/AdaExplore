import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex module that upsamples a 2D feature map, applies RMS normalization across channels,
    then performs 1D Lp-pooling over the flattened spatial dimension and a learned channel-wise gating.
    The final output is a compact (batch_size, out_channels) feature representation.
    """
    def __init__(self, out_channels: int = 32, pool_p: int = 2, pool_kernel: int = 4, eps: float = 1e-6):
        """
        Initializes the model components.

        Args:
            out_channels (int): Number of output channels for the transposed convolution.
            pool_p (int): The p value for Lp pooling (e.g., 2 for Euclidean).
            pool_kernel (int): Kernel size for Lp pooling along the flattened spatial dimension.
            eps (float): Epsilon value for RMSNorm stability.
        """
        super(Model, self).__init__()
        # LazyConvTranspose2d will infer in_channels at first forward call
        self.deconv = nn.LazyConvTranspose2d(
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=True
        )
        # RMSNorm expects normalized_shape equal to channel dimension; we'll permute tensors so last dim = channels
        self.norm = nn.RMSNorm(out_channels, eps=eps, elementwise_affine=True)

        # LPPool1d operates on (N, C, L) where L is a sequence length; we'll flatten spatial dims before pooling
        # Use stride=2 to downsample the flattened spatial dimension
        self.pool = nn.LPPool1d(norm_type=pool_p, kernel_size=pool_kernel, stride=2)

        # Small non-linearity for intermediate activations
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation.

        Pipeline:
        1. Upsample spatially with a transposed convolution (deconv).
        2. Apply GELU activation.
        3. Permute to put channels as the last dimension and apply RMSNorm across channels.
        4. Permute back to (N, C, H, W).
        5. Flatten spatial dims to (N, C, L) and apply Lp pooling (1D).
        6. Compute a channel-wise gating scalar from pooled features and apply it.
        7. Aggregate the gated pooled features into a compact (N, C) representation by mean reduction.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels).
        """
        # 1. Upsample via transposed convolution
        y = self.deconv(x)              # (N, out_channels, H', W')
        # 2. Non-linearity
        y = self.act(y)

        # 3. RMSNorm: permute so channels are the last dimension and normalize across them
        #    RMSNorm expects normalized_shape == num_channels, applied over the last dim(s)
        y = y.permute(0, 2, 3, 1).contiguous()  # (N, H', W', C)
        y = self.norm(y)                        # normalized across C
        y = y.permute(0, 3, 1, 2).contiguous()  # back to (N, C, H', W')

        # 4. Flatten spatial dimensions and apply 1D Lp pooling
        N, C, H, W = y.shape
        z = y.view(N, C, H * W)                 # (N, C, L) where L = H'*W'
        pooled = self.pool(z)                   # (N, C, L2) reduced spatial length

        # 5. Channel-wise gating computed from pooled activations
        gate = torch.sigmoid(torch.mean(pooled, dim=2, keepdim=True))  # (N, C, 1)
        gated = pooled * gate                                          # (N, C, L2)

        # 6. Aggregate spatially to obtain a compact representation per channel
        aggregated = torch.mean(gated, dim=2)   # (N, C)

        return aggregated

# Configuration / default sizes for inputs
batch_size = 8
in_channels = 3
height = 16
width = 16
out_channels = 32
pool_p = 2
pool_kernel = 4

def get_inputs():
    """
    Returns example input tensors matching the expected input signature of the Model.forward.
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for the Model.__init__ in the same order.
    """
    return [out_channels, pool_p, pool_kernel]