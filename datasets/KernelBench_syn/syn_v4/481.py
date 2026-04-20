import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex model that fuses a 2D image tensor with a 1D sequential tensor by:
      1) Zero-padding the sequence (ZeroPad1d),
      2) Reshaping the padded sequence into a spatial map that matches the image HxW,
      3) Concatenating along channels,
      4) Applying 2D average pooling (AvgPool2d),
      5) Applying a Lazy Instance Normalization (LazyInstanceNorm2d) that is lazily initialized
      6) Channel-wise gating using the spatial global average.
    This creates a cross-modal fusion pattern using the requested layers.
    """
    def __init__(
        self,
        img_channels: int,
        seq_channels: int,
        H: int,
        W: int,
        pad_left: int,
        pad_right: int,
        pool_kernel: int,
        pool_stride: int = None,
    ):
        """
        Args:
            img_channels (int): Number of channels in the image tensor.
            seq_channels (int): Number of channels in the sequence tensor.
            H (int): Height of the image spatial map.
            W (int): Width of the image spatial map.
            pad_left (int): Left padding for ZeroPad1d.
            pad_right (int): Right padding for ZeroPad1d.
            pool_kernel (int): Kernel size for AvgPool2d.
            pool_stride (int, optional): Stride for AvgPool2d. If None, defaults to pool_kernel.
        """
        super(Model, self).__init__()
        self.img_channels = img_channels
        self.seq_channels = seq_channels
        self.H = H
        self.W = W
        self.pad_left = pad_left
        self.pad_right = pad_right

        # Pad the 1D sequence on both sides
        self.pad1d = nn.ZeroPad1d((pad_left, pad_right))

        # 2D average pooling to reduce spatial resolution / mix neighborhoods
        self.avgpool = nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_stride)

        # LazyInstanceNorm2d will be initialized on first forward using the concatenated channel count
        self.instnorm = nn.LazyInstanceNorm2d()  # num_features set lazily

    def forward(self, img: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that fuses image and sequence data.

        Args:
            img (torch.Tensor): Image tensor of shape (B, img_channels, H, W).
            seq (torch.Tensor): Sequence tensor of shape (B, seq_channels, L) where
                                L + pad_left + pad_right == H * W (so it can be reshaped to H x W).

        Returns:
            torch.Tensor: Fused tensor after pooling and normalization.
        """
        # 1) Zero-pad the sequence on the temporal/spatial axis
        seq_padded = self.pad1d(seq)  # (B, seq_channels, L + pad_left + pad_right)

        # 2) Reshape padded sequence into a spatial map matching (H, W)
        B, C_seq, Lp = seq_padded.shape
        expected = self.H * self.W
        if Lp != expected:
            # Raise a clear error to help debugging if shapes mismatch
            raise RuntimeError(f"Padded sequence length {Lp} does not match H*W ({expected}).")
        seq_spatial = seq_padded.view(B, C_seq, self.H, self.W)  # (B, seq_channels, H, W)

        # 3) Concatenate image and sequence spatial map along channels
        fused = torch.cat([img, seq_spatial], dim=1)  # (B, img_channels + seq_channels, H, W)

        # 4) Apply average pooling to mix local neighborhoods
        pooled = self.avgpool(fused)  # spatial dims reduced depending on kernel/stride

        # 5) Apply lazy instance normalization (num_features set on first forward)
        normalized = self.instnorm(pooled)

        # 6) Channel-wise gating: compute spatial global average per channel and modulate
        #    This introduces a simple attention-like gating mechanism.
        channel_means = normalized.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        gates = torch.sigmoid(channel_means)  # in (0,1)
        gated = normalized * gates  # (B, C, H', W')

        return gated

# Module-level configuration variables
batch_size = 8
img_channels = 3
seq_channels = 2
H = 16
W = 16
pad_left = 2
pad_right = 2
pool_kernel = 2
pool_stride = 2

# Determine base sequence length before padding so that after padding it reshapes to H*W
base_seq_len = H * W - (pad_left + pad_right)
if base_seq_len <= 0:
    raise ValueError("Configured H and W with pad_left/pad_right produce non-positive base_seq_len.")

def get_inputs():
    """
    Returns example input tensors:
      - img: (batch_size, img_channels, H, W)
      - seq: (batch_size, seq_channels, base_seq_len)
    """
    img = torch.randn(batch_size, img_channels, H, W)
    seq = torch.randn(batch_size, seq_channels, base_seq_len)
    return [img, seq]

def get_init_inputs():
    """
    Returns initialization parameters for the Model constructor in the same order.
    """
    return [img_channels, seq_channels, H, W, pad_left, pad_right, pool_kernel, pool_stride]