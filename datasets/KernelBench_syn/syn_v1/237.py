import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex model that:
    - Applies 3D max pooling to reduce volumetric spatial resolution.
    - Uses LazyInstanceNorm1d across channel dimension for the pooled per-voxel sequences.
    - Treats the pooled spatial locations as a sequence (tgt) for a TransformerDecoder stack,
      using an external memory sequence for cross-attention.
    - Projects decoder outputs back to volumetric shape.

    Inputs:
        x: Tensor of shape (batch_size, in_channels, D, H, W)
        memory: Tensor of shape (mem_seq_len, batch_size, in_channels)  # E == in_channels (d_model)
    """
    def __init__(self, in_channels: int, nhead: int, num_decoder_layers: int, pool_kernel=(2,2,2)):
        """
        Initializes components.

        Args:
            in_channels (int): Number of input channels (also used as d_model for the Transformer).
            nhead (int): Number of attention heads for TransformerDecoderLayer (must divide in_channels).
            num_decoder_layers (int): Number of decoder layers in the TransformerDecoder stack.
            pool_kernel (tuple): Kernel size (and stride) for MaxPool3d to reduce spatial dims.
        """
        super(Model, self).__init__()
        assert in_channels % nhead == 0, "in_channels (d_model) must be divisible by nhead"

        # Reduce spatial size with 3D max pooling
        self.pool = nn.MaxPool3d(kernel_size=pool_kernel, stride=pool_kernel)

        # Lazy instance normalization over channel dimension for sequences of pooled voxels:
        # Expect input to norm.forward to be (batch, channels, seq_len)
        self.inorm = nn.LazyInstanceNorm1d()

        # Transformer decoder stack: uses in_channels as d_model
        decoder_layer = nn.TransformerDecoderLayer(d_model=in_channels, nhead=nhead)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Output projection applied per-channel after decoding (keeps same channel dimension)
        self.out_proj = nn.Linear(in_channels, in_channels)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        Forward computation.

        Args:
            x (torch.Tensor): (batch_size, in_channels, D, H, W)
            memory (torch.Tensor): (mem_seq_len, batch_size, in_channels)

        Returns:
            torch.Tensor: Reconstructed tensor of shape (batch_size, in_channels, D', H', W')
                          where D',H',W' are reduced by pool kernel.
        """
        # 1) Spatial pooling -> (batch, channels, D', H', W')
        pooled = self.pool(x)

        batch, channels, Dp, Hp, Wp = pooled.shape

        # 2) Flatten spatial dimensions to sequence length L = D'*H'*W' -> (batch, channels, L)
        seq_len = Dp * Hp * Wp
        seq = pooled.view(batch, channels, seq_len)

        # 3) Instance-normalize across each channel over the sequence positions (lazy init will infer channels)
        seq_norm = self.inorm(seq)  # (batch, channels, seq_len)

        # 4) Prepare target sequence for TransformerDecoder: (T, N, E) where T=seq_len, N=batch, E=channels
        tgt = seq_norm.permute(2, 0, 1)  # (seq_len, batch, channels)

        # 5) Run through TransformerDecoder with provided memory (mem_seq_len, batch, channels)
        dec_out = self.decoder(tgt, memory)  # (seq_len, batch, channels)

        # 6) Project channel features and reshape back to volumetric grid
        dec_out = dec_out.permute(1, 2, 0)  # (batch, channels, seq_len)
        projected = self.out_proj(dec_out.permute(0, 2, 1)).permute(0, 2, 1)  # (batch, channels, seq_len)
        # Note: .out_proj applied across channel embedding dimension by permuting to (batch, seq_len, channels)
        reconstructed = projected.view(batch, channels, Dp, Hp, Wp)

        return reconstructed

# Configuration variables
batch_size = 4
in_channels = 128  # also used as Transformer d_model
D = 8
H = 8
W = 8
pool_kernel = (2, 2, 2)
nhead = 8
num_decoder_layers = 2
mem_seq_len = 16  # length of external memory sequence used by decoder

def get_inputs():
    """
    Returns:
        x: (batch_size, in_channels, D, H, W)
        memory: (mem_seq_len, batch_size, in_channels)
    """
    x = torch.randn(batch_size, in_channels, D, H, W)
    memory = torch.randn(mem_seq_len, batch_size, in_channels)
    return [x, memory]

def get_init_inputs():
    """
    Returns constructor arguments for Model: [in_channels, nhead, num_decoder_layers]
    """
    return [in_channels, nhead, num_decoder_layers]