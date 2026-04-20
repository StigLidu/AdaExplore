import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    A hybrid 3D-convolutional and sequence-processing model that demonstrates:
      - Lazy initialization for Conv3d and BatchNorm1d (nn.LazyConv3d, nn.LazyBatchNorm1d)
      - Reshaping spatial 3D feature maps into a sequence (N, C, L) for 1D operations
      - Adaptive average pooling over the sequence dimension (nn.AdaptiveAvgPool1d)
      - Combining pooled statistics (mean and max) and a learned projection.

    Forward pass summary:
      x (N, Cin, D, H, W) ->
        conv1 -> ReLU -> conv2 -> ReLU -> reshape to (N, C, L) ->
        LazyBatchNorm1d over C -> AdaptiveAvgPool1d over L to fixed length P ->
        compute mean and max across channel dimension -> concatenate -> linear projection -> output (N, P)
    """
    def __init__(self,
                 out_channels: int,
                 mid_channels: int,
                 kernel_size,
                 pool_output_size: int):
        """
        Initializes the model.

        Args:
            out_channels (int): Number of channels produced by the final convolution.
            mid_channels (int): Number of channels produced by the first convolution.
            kernel_size (int or tuple): Kernel size for the first convolution.
            pool_output_size (int): Fixed output length after AdaptiveAvgPool1d.
        """
        super(Model, self).__init__()
        # Two lazy conv3d layers: in_channels will be inferred on first forward call
        self.conv1 = nn.LazyConv3d(out_channels=mid_channels,
                                   kernel_size=kernel_size,
                                   stride=1,
                                   padding=tuple(k // 2 for k in (kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size))))
        # second conv downsamples two spatial dims to add some complexity
        self.conv2 = nn.LazyConv3d(out_channels=out_channels,
                                   kernel_size=(3, 3, 3),
                                   stride=(1, 2, 2),
                                   padding=(1, 1, 1))

        # LazyBatchNorm1d will infer num_features from incoming channel dimension when called
        self.bn = nn.LazyBatchNorm1d()
        self.pool = nn.AdaptiveAvgPool1d(pool_output_size)
        self.relu = nn.ReLU(inplace=True)

        # Learned projection from combined pooled statistics (2 * pool_output_size -> pool_output_size)
        # We register parameters explicitly because sizes depend on pool_output_size provided at init
        self.pool_output_size = pool_output_size
        proj_out = pool_output_size
        proj_in = pool_output_size * 2
        self.proj_weight = nn.Parameter(torch.randn(proj_out, proj_in) * (1.0 / (proj_in ** 0.5)))
        self.proj_bias = nn.Parameter(torch.zeros(proj_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation.

        Args:
            x (torch.Tensor): Input tensor of shape (N, Cin, D, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (N, pool_output_size)
        """
        # 3D conv feature extraction
        out = self.relu(self.conv1(x))   # (N, mid_channels, D, H, W)
        out = self.relu(self.conv2(out)) # (N, out_channels, D', H', W')

        # Collapse spatial dimensions into a single sequence dimension L = D'*H'*W'
        N, C, Dp, Hp, Wp = out.shape
        seq = out.view(N, C, Dp * Hp * Wp)  # (N, C, L)

        # Normalization across the channel dimension (BatchNorm1d expects (N, C, L) form)
        seq = self.bn(seq)  # LazyBatchNorm1d will initialize num_features=C on first call

        # Reduce / reshape sequence length to a fixed size with adaptive pooling
        seq_pooled = self.pool(seq)  # (N, C, pool_output_size)

        # Compute per-position statistics across channels: mean and max -> shapes (N, pool_output_size)
        mean_feat = seq_pooled.mean(dim=1)         # (N, pool_output_size)
        max_feat, _ = seq_pooled.max(dim=1)        # (N, pool_output_size)

        # Combine statistics and apply a learned linear projection (implemented with explicit params)
        combined = torch.cat([mean_feat, max_feat], dim=1)  # (N, 2*pool_output_size)
        # Use F.linear with our registered parameters (weight shape: out_features x in_features)
        out_final = F.linear(combined, self.proj_weight, self.proj_bias)  # (N, pool_output_size)
        # Final non-linearity
        out_final = torch.tanh(out_final)

        return out_final

# Configuration / default sizes
BATCH_SIZE = 4
IN_CHANNELS = 3
DEPTH = 8
HEIGHT = 32
WIDTH = 32

MID_CHANNELS = 16
OUT_CHANNELS = 32
KERNEL_SIZE = (3, 3, 3)
POOL_OUTPUT_SIZE = 7

def get_inputs():
    """
    Generate a sample input tensor compatible with the model's expected input shape.

    Returns:
        list: [x] where x has shape (BATCH_SIZE, IN_CHANNELS, DEPTH, HEIGHT, WIDTH)
    """
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, DEPTH, HEIGHT, WIDTH, dtype=torch.float32)
    return [x]

def get_init_inputs():
    """
    Return initialization parameters for the Model constructor.

    Returns:
        list: [OUT_CHANNELS, MID_CHANNELS, KERNEL_SIZE, POOL_OUTPUT_SIZE]
    """
    return [OUT_CHANNELS, MID_CHANNELS, KERNEL_SIZE, POOL_OUTPUT_SIZE]