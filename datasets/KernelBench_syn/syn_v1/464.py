import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex model that upsamples an input feature map with ConvTranspose2d,
    processes spatial tokens through a TransformerEncoder, and produces a
    pooled feature vector.

    Computation steps:
    1. ConvTranspose2d upsampling
    2. ReLU activation
    3. Flatten spatial grid to a sequence of tokens (S, N, C)
    4. Linear projection to transformer embedding dim
    5. Add learnable positional encodings
    6. TransformerEncoder processing
    7. Project transformer embeddings back to channel space
    8. Reshape to (N, C, H_out, W_out)
    9. AdaptiveAvgPool2d to produce final pooled output
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        height: int,
        width: int,
        kernel_size: int,
        stride: int,
        padding: int,
        output_padding: int,
        embedding_dim: int,
        nhead: int,
        num_layers: int,
        dropout: float,
        pool_output_size: int,
    ):
        """
        Initializes the components used in the forward pass.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels produced by ConvTranspose2d.
            height: Input height.
            width: Input width.
            kernel_size: Kernel size for ConvTranspose2d.
            stride: Stride for ConvTranspose2d.
            padding: Padding for ConvTranspose2d.
            output_padding: Output padding for ConvTranspose2d.
            embedding_dim: Transformer embedding dimension (d_model).
            nhead: Number of attention heads for TransformerEncoderLayer.
            num_layers: Number of layers in TransformerEncoder.
            dropout: Dropout for TransformerEncoderLayer.
            pool_output_size: Output spatial size for AdaptiveAvgPool2d (int or tuple).
        """
        super(Model, self).__init__()

        # Save geometry
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.height = height
        self.width = width
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

        # ConvTranspose2d to upsample spatial resolution
        self.upconv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=True
        )

        # Activation
        self.act = nn.ReLU(inplace=True)

        # Compute output spatial dimensions after ConvTranspose2d:
        # H_out = (H_in - 1) * stride - 2*padding + kernel_size + output_padding
        H_out = (height - 1) * stride - 2 * padding + kernel_size + output_padding
        W_out = (width - 1) * stride - 2 * padding + kernel_size + output_padding
        self.H_out = H_out
        self.W_out = W_out
        self.seq_len = H_out * W_out

        # Projection layers between channel space and transformer embedding space
        self.input_proj = nn.Linear(out_channels, embedding_dim)
        self.output_proj = nn.Linear(embedding_dim, out_channels)

        # Learnable positional encoding for sequence positions (S, E)
        # Fixed size based on H_out * W_out computed above.
        self.pos_enc = nn.Parameter(torch.randn(self.seq_len, embedding_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, dropout=dropout, batch_first=False)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Adaptive average pooling to aggregate spatial outputs to desired size
        # pool_output_size can be int or tuple; keep it flexible
        self.pool = nn.AdaptiveAvgPool2d(pool_output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            Tensor of shape (batch_size, out_channels, pool_h, pool_w) after adaptive pooling.
            If pool_output_size is 1 (default), this becomes (batch_size, out_channels, 1, 1).
        """
        # 1. Upsample with ConvTranspose2d
        x = self.upconv(x)  # (B, out_channels, H_out, W_out)

        # 2. Activation
        x = self.act(x)

        B, C, H, W = x.shape
        assert H == self.H_out and W == self.W_out, "Unexpected spatial dimensions after ConvTranspose2d."

        # 3. Flatten spatial grid into a sequence: (S, N, C)
        # We'll create shape (B, C, S) then permute
        x_seq = x.view(B, C, H * W).permute(2, 0, 1)  # (S, B, C)

        # 4. Project channel dimension to transformer embedding space: (S, B, E)
        x_emb = self.input_proj(x_seq)  # Linear acts on last dimension

        # 5. Add positional encodings (S, 1, E) broadcast over batch
        pos = self.pos_enc.unsqueeze(1)  # (S, 1, E)
        x_emb = x_emb + pos  # (S, B, E)

        # 6. Transformer encoder (expects (S, N, E))
        x_trans = self.transformer(x_emb)  # (S, B, E)

        # 7. Project embeddings back to channel space: (S, B, C)
        x_chan = self.output_proj(x_trans)

        # 8. Permute and reshape back to spatial map: (B, C, H_out, W_out)
        x_out = x_chan.permute(1, 2, 0).contiguous().view(B, C, H, W)

        # 9. Adaptive average pooling to desired output size
        pooled = self.pool(x_out)  # (B, C, pool_h, pool_w)

        return pooled


# Configuration variables
batch_size = 8
in_channels = 3
out_channels = 48
height = 16
width = 16
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
embedding_dim = 64
nhead = 8
num_layers = 2
dropout = 0.1
pool_output_size = 1  # will produce (B, out_channels, 1, 1)

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Return the initialization parameters in the same order as Model.__init__.
    """
    return [
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        stride,
        padding,
        output_padding,
        embedding_dim,
        nhead,
        num_layers,
        dropout,
        pool_output_size,
    ]