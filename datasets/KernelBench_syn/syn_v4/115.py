import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex fusion module that processes a 1D sequence with a lazy-initialized Conv1d
    and a 2D image with circular padding + Conv2d. The image features are pooled
    and projected to produce channel-wise gates that modulate the sequence features.
    The model demonstrates usage of nn.LazyConv1d, nn.CircularPad2d and nn.LocalResponseNorm
    alongside standard Conv2d, AdaptiveAvgPool2d and Linear layers.

    Forward signature:
        forward(seq: Tensor, img: Tensor) -> Tensor
    Inputs:
        seq: (batch, seq_channels, seq_length)
        img: (batch, img_in_channels, height, width)
    Output:
        out: (batch, conv_out_channels) - fused and projected vector per example
    """
    def __init__(
        self,
        img_in_channels: int,
        conv_out_channels: int = 48,
        pad2d: int = 2,
        img_conv_channels: int = 32,
    ):
        """
        Args:
            img_in_channels (int): Number of channels in the input image tensor.
            conv_out_channels (int): Number of output channels for the 1D conv and final projection.
            pad2d (int): Padding size for CircularPad2d.
            img_conv_channels (int): Number of channels produced by the image Conv2d.
        """
        super(Model, self).__init__()
        # Lazy 1D convolution: in_channels will be inferred on first forward
        self.seq_conv = nn.LazyConv1d(out_channels=conv_out_channels, kernel_size=5, stride=1, padding=2)
        # Local response normalization applied over channels of the sequence conv output
        self.lrn_seq = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0)
        # Circular padding for images to avoid border artifacts
        self.pad2d = nn.CircularPad2d(pad2d)
        # Small 2D conv to extract spatial features from the (padded) image
        self.img_conv = nn.Conv2d(in_channels=img_in_channels, out_channels=img_conv_channels, kernel_size=3, padding=1)
        # Spatial pooling to a single vector per channel
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Project pooled image features into the same channel space as sequence conv output
        # so we can produce channel-wise gating signals
        self.img_proj = nn.Linear(img_conv_channels, conv_out_channels)
        # Final projection from fused channels to output vector (keeps same dim here)
        self.out_proj = nn.Linear(conv_out_channels, conv_out_channels)

    def forward(self, seq: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that fuses sequence and image modalities.

        Steps:
        1. Apply LazyConv1d to sequence + GELU activation.
        2. Apply LocalResponseNorm to stabilize across channels.
        3. Circular-pad the image, apply Conv2d + ReLU, then spatially pool to a single vector.
        4. Project image vector to produce channel-wise gates (sigmoid).
        5. Apply gates to the normalized sequence features and aggregate.
        6. Final linear projection.

        Args:
            seq (torch.Tensor): Sequence tensor of shape (B, C_seq, L).
            img (torch.Tensor): Image tensor of shape (B, C_img, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, conv_out_channels).
        """
        # Sequence branch
        # seq_conv_out: (B, conv_out_channels, L)
        seq_conv_out = self.seq_conv(seq)
        seq_conv_out = F.gelu(seq_conv_out)
        # Normalize across channels
        seq_norm = self.lrn_seq(seq_conv_out)

        # Image branch
        # Circular pad -> conv -> relu -> adaptive pool -> flatten
        padded = self.pad2d(img)
        img_feat = self.img_conv(padded)
        img_feat = F.relu(img_feat)
        pooled = self.adaptive_pool(img_feat).view(img_feat.size(0), -1)  # (B, img_conv_channels)

        # Project image features into channel gating space
        img_proj = F.relu(self.img_proj(pooled))  # (B, conv_out_channels)
        gates = torch.sigmoid(img_proj).unsqueeze(-1)  # (B, conv_out_channels, 1)

        # Apply gates to the normalized sequence features
        gated_seq = seq_norm * gates  # broadcasting along length dimension

        # Aggregate sequence: mean over length to get a per-channel vector
        seq_vec = gated_seq.mean(dim=2)  # (B, conv_out_channels)

        # Residual-style fusion with projected image features
        fused = seq_vec + img_proj  # (B, conv_out_channels)

        # Final projection
        out = self.out_proj(fused)
        out = F.relu(out)
        return out


# Configuration variables for creating inputs
batch_size = 8
seq_channels = 13      # will be lazily inferred by LazyConv1d
seq_length = 1024
img_channels = 3
img_height = 64
img_width = 64

# Model hyperparameters to be passed into get_init_inputs()
conv_out_channels = 48
pad2d = 2
img_conv_channels = 32

def get_inputs():
    """
    Returns a list containing:
      - seq tensor: (batch_size, seq_channels, seq_length)
      - img tensor: (batch_size, img_channels, img_height, img_width)
    """
    seq = torch.randn(batch_size, seq_channels, seq_length)
    img = torch.randn(batch_size, img_channels, img_height, img_width)
    return [seq, img]

def get_init_inputs():
    """
    Returns the initialization arguments for Model.__init__:
      [img_in_channels, conv_out_channels, pad2d, img_conv_channels]
    """
    return [img_channels, conv_out_channels, pad2d, img_conv_channels]