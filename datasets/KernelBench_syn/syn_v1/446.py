import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Volumetric Channel-Attention Mixer

    This module performs a sequence of operations on 5D volumetric input tensors:
      1. 3D average pooling to reduce spatial resolution.
      2. Global spatial average to produce channel descriptors.
      3. A small bottleneck MLP implemented with einsum and learnable parameters:
           descriptor -> hidden (ReLU) -> channel logits
      4. Softmax across channels to obtain per-sample channel attention weights.
      5. Channel-wise re-scaling of the pooled feature map.
      6. A learned cross-channel mixing via an einsum with a mixing matrix.

    Input:
        x (torch.Tensor): shape (B, C, D, H, W)

    Output:
        out (torch.Tensor): shape (B, C, D_out, H_out, W_out)
    """
    def __init__(self, channels: int, pool_kernel: int, pool_stride: int, pool_padding: int, reduction: int = 4):
        """
        Initializes the module.

        Args:
            channels (int): Number of input channels (C).
            pool_kernel (int): Kernel size for AvgPool3d.
            pool_stride (int): Stride for AvgPool3d.
            pool_padding (int): Padding for AvgPool3d.
            reduction (int): Bottleneck reduction factor for the channel MLP.
        """
        super(Model, self).__init__()
        self.channels = channels
        self.avgpool = nn.AvgPool3d(kernel_size=pool_kernel, stride=pool_stride, padding=pool_padding)
        self.relu = nn.ReLU()
        # Softmax over channels
        self.softmax = nn.Softmax(dim=1)

        # Bottleneck sizes
        hidden = max(1, channels // reduction)

        # Bottleneck MLP implemented via parameters and einsum
        # W1: (C, hidden)  -- compute: h_bh = einsum('bc,ch->bh', g, W1)
        # W2: (hidden, C)  -- compute: s_bc = einsum('bh,hc->bc', h, W2)
        self.W1 = nn.Parameter(torch.randn(channels, hidden) * (1.0 / channels**0.5))
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.W2 = nn.Parameter(torch.randn(hidden, channels) * (1.0 / hidden**0.5))
        self.b2 = nn.Parameter(torch.zeros(channels))

        # Cross-channel mixing matrix (C, C)
        self.mix = nn.Parameter(torch.randn(channels, channels) * (1.0 / channels**0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, D, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (B, C, D_out, H_out, W_out)
        """
        # 1) Spatial downsampling
        x_p = self.avgpool(x)  # (B, C, D2, H2, W2)

        # 2) Channel descriptor via global spatial average
        # mean over D2, H2, W2
        g = x_p.mean(dim=[2, 3, 4])  # (B, C)

        # 3) Bottleneck MLP using einsum for projections
        # proj to hidden
        h = torch.einsum('bc,ch->bh', g, self.W1) + self.b1  # (B, hidden)
        h = self.relu(h)  # non-linearity

        # project back to channel logits
        s = torch.einsum('bh,hc->bc', h, self.W2) + self.b2  # (B, C)

        # 4) Channel softmax per sample to get attention weights
        a = self.softmax(s)  # (B, C)

        # 5) Re-scale pooled features by channel attention
        scale = a.view(a.size(0), a.size(1), 1, 1, 1)  # (B, C, 1, 1, 1)
        x_scaled = x_p * scale  # (B, C, D2, H2, W2)

        # 6) Cross-channel mixing: new_channels_k = sum_c x_scaled_c * mix_{c,k}
        out = torch.einsum('bcdhw,ck->bkdhw', x_scaled, self.mix)  # (B, C, D2, H2, W2)

        return out

# Configuration / default sizes
batch_size = 4
channels = 64
depth = 16
height = 32
width = 32
pool_kernel = 2
pool_stride = 2
pool_padding = 0
reduction = 4

def get_inputs():
    """
    Generates a random volumetric input tensor.

    Returns:
        list: [x] where x has shape (batch_size, channels, depth, height, width)
    """
    x = torch.randn(batch_size, channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Initialization parameters for constructing the Model.

    Returns:
        list: [channels, pool_kernel, pool_stride, pool_padding, reduction]
    """
    return [channels, pool_kernel, pool_stride, pool_padding, reduction]