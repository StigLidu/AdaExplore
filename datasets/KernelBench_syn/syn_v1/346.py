import torch
import torch.nn as nn

# Configuration
BATCH_SIZE = 8
IN_CHANNELS = 64
H = 16
W = 16
PAD = 1  # circular padding on each side
POOL_KERNEL = 3
POOL_STRIDE = 2
OUT_FEATURES = 128
DTYPE = torch.float32
DEVICE = torch.device("cpu")


class Model(nn.Module):
    """
    Complex module that demonstrates combining circular 2D padding, 1D average pooling,
    Mish activation, sequence modulation, and a learned linear projection.

    Input:
        images: Tensor of shape (N, C, H, W)
        context: Tensor of shape (N, L_pooled) where L_pooled is computed from H, W, PAD, POOL_KERNEL, POOL_STRIDE

    Computation steps:
        1. Circular 2D padding to preserve wrap-around spatial context.
        2. Collapse spatial dims (H_padded, W_padded) into a single sequence dimension L.
        3. Apply AvgPool1d over the sequence to downsample (kernel + stride).
        4. Apply Mish nonlinearity.
        5. Modulate sequence tokens by per-position context weights (element-wise).
        6. Linearly project the channel dimension to OUT_FEATURES via a learned matrix.
        7. Global-average pool over sequence length to produce (N, OUT_FEATURES).
    """
    def __init__(self):
        super(Model, self).__init__()
        # Circular pad for 2D inputs
        self.pad2d = nn.CircularPad2d(PAD)
        # AvgPool1d to downsample the flattened spatial sequence
        self.pool1d = nn.AvgPool1d(kernel_size=POOL_KERNEL, stride=POOL_STRIDE)
        # Mish activation
        self.mish = nn.Mish()
        # Learned linear projection from IN_CHANNELS -> OUT_FEATURES
        # We'll implement it as a parameter matrix multiplied on the last dimension.
        self.proj = nn.Parameter(torch.randn(IN_CHANNELS, OUT_FEATURES, dtype=DTYPE))
        # Optional bias for projection
        self.proj_bias = nn.Parameter(torch.zeros(OUT_FEATURES, dtype=DTYPE))

    def forward(self, images: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            images: (N, C, H, W)
            context: (N, L_pooled) per-position modulation vector

        Returns:
            Tensor of shape (N, OUT_FEATURES)
        """
        # 1) Circular pad the 2D image -> (N, C, H_padded, W_padded)
        x = self.pad2d(images)

        # 2) Collapse spatial dims into a sequence dimension L = H_padded * W_padded
        N, C, H_p, W_p = x.shape
        x = x.reshape(N, C, H_p * W_p)  # (N, C, L)

        # 3) Apply 1D average pooling along the sequence dimension -> (N, C, L_pooled)
        x = self.pool1d(x)

        # 4) Nonlinearity
        x = self.mish(x)  # (N, C, L_pooled)

        # 5) Modulate each sequence position by the context vector
        # context: (N, L_pooled) -> expand to (N, L_pooled, 1)
        # we want to multiply per-position across channels, so transpose x
        x = x.permute(0, 2, 1)  # (N, L_pooled, C)
        context_exp = context.unsqueeze(-1)  # (N, L_pooled, 1)
        x = x * context_exp  # (N, L_pooled, C) element-wise modulation

        # 6) Linear projection on the channel dimension: (N, L_pooled, C) @ (C, OUT_FEATURES) -> (N, L_pooled, OUT_FEATURES)
        x = torch.matmul(x, self.proj) + self.proj_bias  # (N, L_pooled, OUT_FEATURES)

        # 7) Aggregate across sequence (global average pooling over L_pooled) -> (N, OUT_FEATURES)
        x = x.mean(dim=1)

        return x


# Precompute derived sizes for input generation
H_PAD = H + 2 * PAD
W_PAD = W + 2 * PAD
L = H_PAD * W_PAD
L_POOLED = (L - POOL_KERNEL) // POOL_STRIDE + 1  # floor division consistent with nn.AvgPool1d

def get_inputs():
    """
    Returns:
        [images, context] where:
            images: Tensor of shape (BATCH_SIZE, IN_CHANNELS, H, W)
            context: Tensor of shape (BATCH_SIZE, L_POOLED)
    """
    images = torch.randn(BATCH_SIZE, IN_CHANNELS, H, W, dtype=DTYPE, device=DEVICE)
    # context values around 1.0 (so modulation scales around identity) but random
    context = torch.randn(BATCH_SIZE, L_POOLED, dtype=DTYPE, device=DEVICE) * 0.5 + 1.0
    return [images, context]


def get_init_inputs():
    """
    No external initialization inputs required; the module has internal learnable parameters.
    """
    return []