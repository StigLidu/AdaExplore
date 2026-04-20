import torch
import torch.nn as nn

# Configuration variables
batch_size = 8
in_channels = 3
embed_dim = 64
height = 32
width = 32

pool_kernel = 2
pool_stride = 2

nhead = 8
dim_feedforward = 256
dropout = 0.1

class Model(nn.Module):
    """
    Complex model combining convolutional projection, spatial pooling/unpooling,
    a Transformer encoder layer operating over spatial tokens, and a LogSigmoid activation.
    The model demonstrates interaction between CNN-style spatial ops and a Transformer
    applied to flattened spatial tokens.
    """
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        pool_kernel: int = 2,
        pool_stride: int = 2,
        nhead: int = 8,
        dim_feedforward: int = 256,
        dropout: float = 0.1
    ):
        """
        Initializes the model components.

        Args:
            in_channels (int): Number of input channels.
            embed_dim (int): Embedding dimension (channels) used for transformer tokens.
            pool_kernel (int): Kernel size for max pooling.
            pool_stride (int): Stride for max pooling.
            nhead (int): Number of attention heads in the Transformer encoder layer.
            dim_feedforward (int): Feedforward network dimension in Transformer encoder.
            dropout (float): Dropout probability in Transformer encoder.
        """
        super(Model, self).__init__()

        # Project input channels to embedding channels for tokenization
        self.conv_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)

        # MaxPool2d to reduce spatial resolution and produce indices for unpooling
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride, return_indices=True)

        # MaxUnpool2d to invert the pooling using the indices
        self.unpool = nn.MaxUnpool2d(kernel_size=pool_kernel, stride=pool_stride)

        # Transformer encoder layer operates on (sequence_length, batch_size, embed_dim)
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=False  # we'll provide (S, N, E)
        )

        # Project back to original number of channels
        self.conv_back = nn.Conv2d(embed_dim, in_channels, kernel_size=1)

        # Element-wise log-sigmoid activation
        self.logsigmoid = nn.LogSigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
         - Convolutional projection to embedding channels
         - Max pooling (returns indices)
         - Flatten spatial dims into a sequence and run through TransformerEncoderLayer
         - Reshape transformer output back to pooled spatial grid
         - Max unpool using stored indices to restore original spatial resolution
         - Residual fuse with pre-pooled projection and project back to input channels
         - Apply LogSigmoid activation

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, H, W)

        Returns:
            torch.Tensor: Output tensor of same spatial shape as input with in_channels channels.
        """
        # Project input to embeddings
        feat = self.conv_proj(x)  # (N, E, H, W)

        # Store shape for unpool output sizing
        prepool_shape = feat.shape

        # Max pool -> get indices for unpool
        pooled, indices = self.pool(feat)  # pooled: (N, E, H//2, W//2) if stride=2

        N, E, Hp, Wp = pooled.shape

        # Flatten spatial dims into sequence tokens for transformer
        # pooled_seq: (S, N, E) where S = Hp * Wp
        pooled_seq = pooled.reshape(N, E, Hp * Wp).permute(2, 0, 1)

        # Transformer encoder layer
        transformed_seq = self.transformer_layer(pooled_seq)  # (S, N, E)

        # Reshape back to (N, E, Hp, Wp)
        transformed = transformed_seq.permute(1, 2, 0).reshape(N, E, Hp, Wp)

        # Unpool to get back to original spatial resolution using indices and output_size
        unpooled = self.unpool(transformed, indices, output_size=prepool_shape)  # (N, E, H, W)

        # Residual fusion with original projection before pooling
        fused = unpooled + feat  # (N, E, H, W)

        # Project back to input channels
        out = self.conv_back(fused)  # (N, in_channels, H, W)

        # Final non-linearity
        out = self.logsigmoid(out)

        return out

# Input generation sizes
def get_inputs():
    """
    Generates a random input tensor suitable for the model.

    Returns:
        list: A list containing a single input tensor of shape (batch_size, in_channels, height, width)
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters in order used by the Model constructor.

    Returns:
        list: [in_channels, embed_dim, pool_kernel, pool_stride, nhead, dim_feedforward, dropout]
    """
    return [in_channels, embed_dim, pool_kernel, pool_stride, nhead, dim_feedforward, dropout]