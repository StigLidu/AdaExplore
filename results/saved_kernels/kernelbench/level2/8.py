import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Optimized ModelNew with three focused improvements:
      1) Fold the fixed scalar division into conv parameters at init (no per-forward scaling).
      2) Fold the small post-pool bias into conv.bias at init (removes broadcast/add).
      3) Use channels_last_3d and autocast to FP16 in forward to utilize Tensor Cores on Ampere GPUs.

    Pipeline after init:
      - Convolution (nn.Conv3d) [weights already scaled; bias contains folded post-bias]
      - MaxPool3d (kernel_size = pool_size, stride = pool_size)
      - AdaptiveAvgPool3d((1,1,1))
      - Sum over the specified dimension (sum_dim)

    The output shape remains (B, 1, 1, 1) for the provided configuration.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        # Keep conv in PyTorch (highly optimized)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)

        # Pool layers using standard PyTorch implementations
        self.max_pool = nn.MaxPool3d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Keep sum dim
        self.sum_dim = sum_dim

        # Enable cudnn autotuner for better conv algorithms on repeated sizes
        torch.backends.cudnn.benchmark = True

        # Fold divisor and the tiny post-pool bias into conv parameters once at init.
        # Create the small post-pool bias tensor (same initialization as before) and fold it into conv.bias.
        # We do all modifications under torch.no_grad() so parameters remain properly registered.
        init_post_bias = torch.randn(bias_shape, dtype=torch.float32)  # shape (C,1,1,1)
        with torch.no_grad():
            # Ensure conv has a bias parameter to fold into
            if self.conv.bias is None:
                self.conv.bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.float32))

            # Fold the fixed divisor into conv weights/bias so forward does not need to divide.
            inv_div = 1.0 / float(divisor)
            self.conv.weight.data.mul_(inv_div)
            self.conv.bias.data.mul_(inv_div)

            # Fold the tiny post-pool bias (per-channel) into conv.bias.
            # init_post_bias.view(-1) has length out_channels.
            self.conv.bias.data.add_(init_post_bias.view(-1))

    def forward(self, x):
        # x: (B, in_channels, D, H, W)
        # Use channels-last-3d memory format for better memory access on Ampere.
        x = x.contiguous(memory_format=torch.channels_last_3d)

        # Use mixed precision to utilize Tensor Cores; conv parameters remain FP32.
        # This provides faster conv + pooling on Ampere while keeping numerical stability of params.
        with torch.cuda.amp.autocast(dtype=torch.float16):
            conv_out = self.conv(x)                         # (B, C, D, H, W)
            pooled = self.max_pool(conv_out)                # (B, C, D', H', W')
            avg_pooled = self.global_avg_pool(pooled)       # (B, C, 1, 1, 1)

            # Sum along the specified dimension (e.g., dim=1 for channels)
            result = torch.sum(avg_pooled, dim=self.sum_dim)  # (B, 1, 1, 1)

        return result