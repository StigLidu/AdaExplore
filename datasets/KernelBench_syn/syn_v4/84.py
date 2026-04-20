import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Configuration
batch_size = 2
latent_channels = 8
latent_depth = 4
latent_height = 4
latent_width = 4

d_model = 64               # Transformer embedding dim (must be divisible by nhead)
nhead = 8
num_decoder_layers = 3

up_out_channels = 32       # output channels of the LazyConvTranspose3d
conv_kernel_size = 2       # use kernel_size=2, stride=2 to double each spatial dimension
conv_stride = 2

mem_len = 4                # length of the memory sequence provided to the decoder
final_out_channels = 16    # number of output channels after reconstruction


def _sinusoidal_positional_encoding(seq_len: int, dim: int, device: torch.device):
    """
    Create a sinusoidal positional encoding of shape (seq_len, dim).
    """
    pe = torch.zeros(seq_len, dim, device=device)
    position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float, device=device) *
                         -(math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class Model(nn.Module):
    """
    Complex module combining a lazy 3D transposed convolution for upsampling,
    instance normalization (applied on the spatial 2D slices), and a TransformerDecoder
    operating on flattened spatial patches. The pipeline:

    1. Upsample a latent 3D volume with LazyConvTranspose3d (in_channels determined lazily).
    2. Collapse the depth dimension by global averaging to form 2D feature maps.
    3. Apply InstanceNorm2d.
    4. Project spatial feature maps to d_model via 1x1 Conv2d, flatten to a sequence.
    5. Build a short learned memory sequence derived from global pooled features.
    6. Add sinusoidal positional encodings and feed into TransformerDecoder.
    7. Reconstruct output feature maps and return a final tensor.
    """
    def __init__(
        self,
        d_model: int = d_model,
        nhead: int = nhead,
        num_decoder_layers: int = num_decoder_layers,
        up_out_channels: int = up_out_channels,
        kernel_size: int = conv_kernel_size,
        stride: int = conv_stride,
        mem_len: int = mem_len,
        final_out_channels: int = final_out_channels,
    ):
        super(Model, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_decoder_layers = num_decoder_layers
        self.up_out_channels = up_out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.mem_len = mem_len
        self.final_out_channels = final_out_channels

        # LazyConvTranspose3d: in_channels is inferred on first forward
        self.upconv3d = nn.LazyConvTranspose3d(
            out_channels=self.up_out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            bias=True
        )

        # InstanceNorm2d applied on the 2D feature maps after collapsing depth
        self.inst_norm2d = nn.InstanceNorm2d(num_features=self.up_out_channels, affine=True)

        # Project from up_out_channels -> d_model per spatial location (1x1 conv)
        self.proj_to_dmodel = nn.Conv2d(self.up_out_channels, self.d_model, kernel_size=1)

        # Transformer decoder stack
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model, nhead=self.nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.num_decoder_layers)

        # Map decoder outputs back to a spatial channel dimension
        self.project_back = nn.Conv2d(self.d_model, self.up_out_channels, kernel_size=1)

        # Final small conv to produce desired output channels (e.g., reconstruction)
        self.final_conv = nn.Conv2d(self.up_out_channels, self.final_out_channels, kernel_size=3, padding=1)

        # Small MLP to transform pooled vector into memory embeddings
        self.mem_mlp = nn.Sequential(
            nn.Linear(self.up_out_channels, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Latent 3D tensor of shape (B, C_latent, D, H, W)

        Returns:
            Tensor of shape (B, final_out_channels, H_out, W_out)
        """
        # 1) Upsample 3D latent volume
        # Output shape: (B, up_out_channels, D_out, H_out, W_out)
        y = self.upconv3d(x)

        # 2) Collapse depth dimension by averaging -> (B, C, H, W)
        y2d = y.mean(dim=2)  # average over depth

        # 3) InstanceNorm2d
        y_norm = self.inst_norm2d(y2d)

        # 4) Project to d_model and flatten spatial dims to sequence
        # y_proj shape: (B, d_model, H_out, W_out)
        y_proj = self.proj_to_dmodel(y_norm)
        B, Dm, H, W = y_proj.shape
        seq_len = H * W

        # tgt for decoder: shape (seq_len, B, d_model)
        tgt = y_proj.view(B, Dm, seq_len).permute(2, 0, 1).contiguous()

        # 5) Build memory: derive a pooled vector per batch and expand to mem_len
        pooled = y_norm.mean(dim=[2, 3])  # (B, up_out_channels)
        mem_embed = self.mem_mlp(pooled)  # (B, d_model)
        # memory shape (mem_len, B, d_model)
        memory = mem_embed.unsqueeze(0).repeat(self.mem_len, 1, 1)

        # 6) Add sinusoidal positional encodings to both tgt and memory
        device = tgt.device
        pos_tgt = _sinusoidal_positional_encoding(seq_len, self.d_model, device)  # (seq_len, d_model)
        pos_mem = _sinusoidal_positional_encoding(self.mem_len, self.d_model, device)  # (mem_len, d_model)

        # broadcast positional encodings: add to each batch
        tgt = tgt + pos_tgt.unsqueeze(1)
        memory = memory + pos_mem.unsqueeze(1)

        # 7) Transformer decoding
        # output shape: (seq_len, B, d_model)
        decoded = self.transformer_decoder(tgt=tgt, memory=memory)

        # 8) Bring back to spatial layout
        decoded = decoded.permute(1, 2, 0).contiguous().view(B, self.d_model, H, W)  # (B, d_model, H, W)

        # 9) Project back to convolutional channels and finalize
        out_conv = self.project_back(decoded)  # (B, up_out_channels, H, W)
        out = self.final_conv(out_conv)       # (B, final_out_channels, H, W)

        return out


def get_inputs():
    """
    Returns a list containing a single latent 3D tensor input:
    shape (batch_size, latent_channels, latent_depth, latent_height, latent_width)
    """
    x = torch.randn(batch_size, latent_channels, latent_depth, latent_height, latent_width)
    return [x]


def get_init_inputs():
    """
    Return the initialization parameters for the Model so callers may instantiate it similarly.
    Order matches Model.__init__ signature defaults.
    """
    return [d_model, nhead, num_decoder_layers, up_out_channels, conv_kernel_size, conv_stride, mem_len, final_out_channels]