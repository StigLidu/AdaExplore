import torch
import torch.nn as nn

"""
Complex example combining a TransformerDecoderLayer with a 3D transposed convolution
and a final Softmax over channels. The model processes a sequence of embeddings
(with length equal to depth*height*width), refines them with a Transformer decoder
(using a projected version of the sequence as "memory"), projects the sequence to
3D feature maps, upsamples / decodes them with ConvTranspose3d, and produces a
channel-normalized output via Softmax.

Structure:
- Model(nn.Module)
  - __init__: constructs TransformerDecoderLayer, projection layers, ConvTranspose3d, activations.
  - forward: applies the described pipeline.
- get_inputs(): returns a random input tensor shaped (batch_size, seq_len, embed_dim)
- get_init_inputs(): returns the initialization parameters for the Model constructor

Configuration variables are defined at module level for easy adjustments.
"""

# Configuration (module-level)
batch_size = 8
depth = 4
height = 4
width = 4
embed_dim = 64
nhead = 8
dim_feedforward = 256

conv_in_channels = 32      # number of channels after linear projection before ConvTranspose3d
conv_out_channels = 16     # final number of output channels (e.g., classes or modalities)

# ConvTranspose3d parameters (will expand spatial dimensions depending on stride)
kernel_size = 3
stride = 2
padding = 1

# Derived sequence length
seq_len = depth * height * width


class Model(nn.Module):
    """
    Model that:
    - Accepts a sequence of embeddings shaped (batch_size, seq_len, embed_dim),
      where seq_len == depth*height*width.
    - Uses a TransformerDecoderLayer to perform attention-based refinement where
      memory is a learned linear projection of the same sequence.
    - Projects the refined sequence into conv_in_channels and reshapes into a 3D
      tensor of shape (batch_size, conv_in_channels, depth, height, width).
    - Applies ConvTranspose3d to decode / upsample the 3D features into
      conv_out_channels.
    - Applies ReLU then channel-wise Softmax (dim=1) to produce normalized outputs.
    """

    def __init__(
        self,
        embed_dim: int,
        nhead: int,
        dim_feedforward: int,
        conv_in_channels: int,
        conv_out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        depth: int,
        height: int,
        width: int,
    ):
        """
        Initializes the model components.

        Args:
            embed_dim: dimensionality of the input embeddings (d_model for decoder).
            nhead: number of attention heads for the TransformerDecoderLayer.
            dim_feedforward: hidden size for the decoder feedforward network.
            conv_in_channels: number of channels to project the sequence into before ConvTranspose3d.
            conv_out_channels: number of channels output by ConvTranspose3d.
            kernel_size, stride, padding: ConvTranspose3d kernel/stride/padding.
            depth, height, width: spatial dimensions to reshape the sequence into.
        """
        super(Model, self).__init__()

        self.embed_dim = embed_dim
        self.depth = depth
        self.height = height
        self.width = width
        self.expected_seq_len = depth * height * width
        self.conv_in_channels = conv_in_channels

        # TransformerDecoderLayer expects input shaped (seq_len, batch, embed_dim)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward
        )

        # Project input sequence to create a "memory" tensor for the decoder (same shape semantics)
        self.memory_proj = nn.Linear(embed_dim, embed_dim)

        # Project decoder outputs to conv_in_channels before reshaping into 3D
        self.post_proj = nn.Linear(embed_dim, conv_in_channels)

        # 3D transposed convolution to decode / upsample the 3D feature volume
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels=conv_in_channels,
            out_channels=conv_out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        # Activation and final normalization over channels
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)

        # Optional small layer norm applied after transformer for stability
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim) where
               seq_len must equal depth * height * width.

        Returns:
            Tensor of shape (batch_size, conv_out_channels, out_depth, out_height, out_width),
            with a Softmax applied across the channel dimension.
        """
        batch, seq_len_in, emb = x.shape
        if seq_len_in != self.expected_seq_len:
            raise ValueError(
                f"Input sequence length ({seq_len_in}) does not match expected "
                f"depth*height*width ({self.expected_seq_len})."
            )
        if emb != self.embed_dim:
            raise ValueError(
                f"Input embedding dim ({emb}) does not match model embed_dim ({self.embed_dim})."
            )

        # Prepare for TransformerDecoderLayer which expects (seq_len, batch, embed_dim)
        # tgt is the sequence to be decoded, memory is a projection of the same sequence.
        tgt = x.permute(1, 0, 2)  # (seq_len, batch, embed_dim)
        memory = self.memory_proj(tgt)  # (seq_len, batch, embed_dim)

        # Optionally normalize before feeding to decoder
        tgt = self.layer_norm(tgt)

        # Transformer decoding (self-attention + cross-attention to memory + feedforward)
        decoded = self.decoder_layer(tgt, memory)  # (seq_len, batch, embed_dim)

        # Bring back to (batch, seq_len, embed_dim)
        decoded = decoded.permute(1, 0, 2).contiguous()  # (batch, seq_len, embed_dim)

        # Project final embeddings to conv_in_channels and reshape to 3D spatial volume
        projected = self.post_proj(decoded)  # (batch, seq_len, conv_in_channels)
        # Reshape sequence dimension into (depth, height, width)
        projected = projected.view(batch, self.conv_in_channels, self.depth, self.height, self.width)

        # Apply ConvTranspose3d to decode/upsample the volume
        out = self.conv_transpose3d(projected)  # (batch, conv_out_channels, outD, outH, outW)

        # Non-linearity and channel-wise Softmax
        out = self.relu(out)
        out = self.softmax(out)  # normalize across channels

        return out


def get_inputs():
    """
    Generates a randomized input sequence of shape (batch_size, seq_len, embed_dim).

    seq_len is derived from depth*height*width.
    """
    x = torch.randn(batch_size, seq_len, embed_dim)
    return [x]


def get_init_inputs():
    """
    Returns the initialization arguments for the Model constructor in order.
    """
    return [
        embed_dim,
        nhead,
        dim_feedforward,
        conv_in_channels,
        conv_out_channels,
        kernel_size,
        stride,
        padding,
        depth,
        height,
        width,
    ]