import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex module that:
    - Upsamples an input image via nearest-neighbor upsampling
    - Projects image patches into a d_model embedding space
    - Uses a TransformerDecoderLayer to attend image patches to an external memory sequence
    - Applies a SiLU activation and projects back to image channel space
    - Adds a residual connection to the upsampled image

    Forward signature:
        forward(image: Tensor, memory: Tensor) -> Tensor

    Args:
        d_model (int): embedding dimension for transformer and projections
        nhead (int): number of heads in multi-head attention
        dim_feedforward (int): hidden dimension in transformer's feedforward net
        in_channels (int): number of channels in the input image
        scale_factor (int or tuple): upsampling scale factor for spatial dims
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        in_channels: int = 3,
        scale_factor: int = 2,
    ):
        super(Model, self).__init__()
        self.d_model = d_model
        self.in_channels = in_channels
        self.scale_factor = scale_factor

        # Upsampling (Nearest Neighbor)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=scale_factor)

        # Project per-pixel/channel features into d_model embeddings and back
        # We'll treat each spatial location as a token: linear maps C -> d_model and back
        self.input_proj = nn.Linear(in_channels, d_model)
        self.output_proj = nn.Linear(d_model, in_channels)

        # Transformer decoder layer to let image tokens attend to external memory
        # Uses default dropout and activation inside; activation here outside is SiLU
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation='relu',  # internal FF activation; additional SiLU applied later
            batch_first=False   # we'll use (seq_len, batch, embed_dim)
        )

        # Non-linear activation applied after transformer output
        self.activation = nn.SiLU()

        # Small normalization to stabilize projection outputs
        self.norm = nn.LayerNorm(d_model)

    def forward(self, image: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Tensor of shape (B, C, H, W)
            memory: External memory sequence; accepted shapes:
                - (B, S, d_model)  OR
                - (S, B, d_model)
                where S is memory sequence length. If given as (B, S, d_model),
                it will be transposed to (S, B, d_model) for the transformer.

        Returns:
            Tensor of shape (B, C, H * scale_factor, W * scale_factor)
        """
        B, C, H, W = image.shape
        # 1) Upsample spatial resolution
        up = self.upsample(image)  # (B, C, H2, W2)
        _, _, H2, W2 = up.shape

        # 2) Flatten spatial dims into a sequence of tokens:
        #    (B, C, H2, W2) -> (B, C, T) -> (B, T, C)
        T = H2 * W2
        x = up.view(B, C, T).permute(0, 2, 1)  # (B, T, C)

        # 3) Project channels to d_model: (B, T, C) -> (B, T, d_model)
        proj = self.input_proj(x)  # (B, T, d_model)

        # 4) Prepare tgt and memory for TransformerDecoderLayer: expects (T, B, E) and (S, B, E)
        tgt = proj.permute(1, 0, 2)  # (T, B, d_model)

        # Accept memory in either (B, S, E) or (S, B, E)
        if memory.dim() == 3 and memory.shape[0] == B:
            # (B, S, E) -> (S, B, E)
            memory_seq = memory.permute(1, 0, 2)
        else:
            memory_seq = memory  # assume already (S, B, E)

        # 5) Transformer decoding: image tokens attend to memory
        dec_out = self.decoder_layer(tgt, memory_seq)  # (T, B, d_model)

        # 6) Back to (B, T, d_model), normalize and apply SiLU
        dec_out = dec_out.permute(1, 0, 2)  # (B, T, d_model)
        dec_out = self.norm(dec_out)
        activated = self.activation(dec_out)  # (B, T, d_model)

        # 7) Project back to channel space: (B, T, d_model) -> (B, T, C)
        recon = self.output_proj(activated)  # (B, T, C)

        # 8) Reshape to spatial map and add residual with upsampled image
        recon = recon.permute(0, 2, 1).contiguous().view(B, C, H2, W2)  # (B, C, H2, W2)
        out = recon + up  # residual connection

        return out

# Module-level configuration variables
batch_size = 4
in_channels = 3
H = 16
W = 16
scale_factor = 2

d_model = 128
nhead = 8
dim_feedforward = 512
memory_seq_len = 64  # length of external memory sequence

def get_inputs():
    """
    Returns:
        [image, memory] where:
          - image: Tensor of shape (B, C, H, W)
          - memory: Tensor of shape (B, S, d_model) (will be converted internally)
    """
    image = torch.randn(batch_size, in_channels, H, W)
    # Memory provided in shape (B, S, d_model) for convenience
    memory = torch.randn(batch_size, memory_seq_len, d_model)
    return [image, memory]

def get_init_inputs():
    """
    Returns the initialization parameters for the Model class:
        [d_model, nhead, dim_feedforward, in_channels, scale_factor]
    """
    return [d_model, nhead, dim_feedforward, in_channels, scale_factor]