import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex model that combines 3D instance normalization (lazy),
    a projection to a transformer embedding, a TransformerDecoderLayer,
    and nonlinear activation (Softplus). The spatial dimensions are
    flattened into a sequence for the transformer decoder, and the
    decoded sequence is projected back to a 5D tensor output.
    """
    def __init__(
        self,
        in_channels: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1
    ):
        """
        Initializes the model components.

        Args:
            in_channels (int): Number of channels in the input 3D tensor.
            d_model (int): Embedding dimension for the Transformer decoder.
            nhead (int): Number of attention heads in the Transformer decoder.
            dim_feedforward (int): Dimension of the feedforward network in the decoder.
            dropout (float): Dropout probability for the decoder layer.
        """
        super(Model, self).__init__()

        # Lazy instance normalization will infer num_features when first forward pass happens
        self.inorm = nn.LazyInstanceNorm3d()

        # Project from input channels (per spatial location) to transformer embedding (conv1d over sequence)
        self.proj_to_emb = nn.Conv1d(in_channels=in_channels, out_channels=d_model, kernel_size=1)

        # Transformer decoder layer: operates on (seq_len, batch, d_model)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu'
        )

        # Non-linear activation
        self.softplus = nn.Softplus()

        # Project back from embedding dimension to output channels (per spatial location)
        self.proj_from_emb = nn.Conv1d(in_channels=d_model, out_channels=in_channels, kernel_size=1)

        # small learnable scaling for residual connection
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).
            memory (torch.Tensor): Memory tensor for the TransformerDecoderLayer of shape
                                   (mem_seq_len, batch_size, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, in_channels, depth, height, width).
        """
        # x: (B, C, D, H, W)
        b, c, d, h, w = x.shape

        # Instance normalization (lazy) -> same shape
        x_norm = self.inorm(x)

        # Non-linear activation
        x_act = self.softplus(x_norm)

        # Flatten spatial dimensions into a sequence: (B, C, L) where L = D*H*W
        L = d * h * w
        x_flat = x_act.view(b, c, L)  # (B, C, L)

        # Project each spatial location's channel vector into transformer embedding:
        # conv1d expects (B, in_channels, seq_len) and returns (B, d_model, seq_len)
        x_emb = self.proj_to_emb(x_flat)  # (B, d_model, L)

        # Prepare target for transformer: (seq_len, batch, d_model)
        tgt = x_emb.permute(2, 0, 1).contiguous()  # (L, B, d_model)

        # memory is expected to be (mem_seq_len, batch, d_model)
        # Apply transformer decoder layer
        dec_out = self.decoder_layer(tgt=tgt, memory=memory)  # (L, B, d_model)

        # Non-linearity after decoder
        dec_act = self.softplus(dec_out)

        # Residual connection (scaled) from original embedding into decoded output
        dec_res = dec_act + self.res_scale * tgt  # (L, B, d_model)

        # Bring back to (B, d_model, L) for projection
        dec_res_b = dec_res.permute(1, 2, 0).contiguous()  # (B, d_model, L)

        # Project back to original channel dimension per spatial location
        out_flat = self.proj_from_emb(dec_res_b)  # (B, C, L)

        # Reshape back to 5D tensor (B, C, D, H, W)
        out = out_flat.view(b, c, d, h, w)

        return out

# Configuration / default parameters
batch_size = 8
in_channels = 16
depth = 8
height = 8
width = 8

d_model = 32
nhead = 4
dim_feedforward = 128
dropout = 0.1

mem_seq_len = 10  # length of the memory sequence provided to the decoder

def get_inputs():
    """
    Returns the input tensors for a forward pass:
      - x: a 5D tensor of shape (batch_size, in_channels, depth, height, width)
      - memory: a memory tensor for the Transformer decoder of shape (mem_seq_len, batch_size, d_model)
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    memory = torch.randn(mem_seq_len, batch_size, d_model)
    return [x, memory]

def get_init_inputs():
    """
    Returns the initialization parameters in the same order expected by Model.__init__:
      - in_channels, d_model, nhead, dim_feedforward, dropout
    """
    return [in_channels, d_model, nhead, dim_feedforward, dropout]