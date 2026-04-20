import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex 3D-volume to sequence Transformer module that:
    - Pads a 5D volumetric input with ConstantPad3d
    - Flattens spatial dimensions into a sequence of tokens
    - Projects channel features into a model embedding space
    - Applies a MultiheadAttention self-attention block with residual+LayerNorm
    - Passes tokens through a TransformerEncoder stack
    - Projects tokens back to channel space and reshapes to padded volume shape
    """
    def __init__(
        self,
        in_channels: int,
        model_dim: int,
        nhead: int,
        num_layers: int,
        padding,                 # int or 6-tuple for ConstantPad3d
        pad_value: float = 0.0
    ):
        """
        Args:
            in_channels (int): Number of input channels/features per voxel.
            model_dim (int): Embedding dimension for attention/transformer.
            nhead (int): Number of attention heads (must divide model_dim).
            num_layers (int): Number of TransformerEncoder layers.
            padding (int or tuple): Padding for ConstantPad3d (int or 6-tuple).
            pad_value (float): Constant value for padding.
        """
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.model_dim = model_dim
        self.nhead = nhead
        self.num_layers = num_layers
        self.padding = padding
        self.pad_value = pad_value

        # Pad 3D volumes (N, C, D, H, W)
        self.pad = nn.ConstantPad3d(self.padding, self.pad_value)

        # Project channel features to transformer embedding and back
        self.input_proj = nn.Linear(in_channels, model_dim)
        self.output_proj = nn.Linear(model_dim, in_channels)

        # Self-attention layer (operates on sequence of tokens)
        self.mha = nn.MultiheadAttention(embed_dim=model_dim, num_heads=nhead)

        # Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Layer normalization applied after the first attention residual
        self.post_attn_ln = nn.LayerNorm(model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, in_channels, padded_depth, padded_height, padded_width).
        """
        # 1) Pad the volumetric input
        #    x_pad: (B, C, D', H', W')
        x_pad = self.pad(x)

        B, C, Dp, Hp, Wp = x_pad.shape

        # 2) Flatten spatial dims into a sequence of tokens:
        #    tokens shape -> (seq_len, B, C) where seq_len = D'*H'*W'
        seq_len = Dp * Hp * Wp
        tokens = x_pad.reshape(B, C, seq_len).permute(2, 0, 1)  # (seq_len, B, C)

        # 3) Project channel features to embedding dimension for transformer
        #    proj_tokens: (seq_len, B, model_dim)
        proj_tokens = self.input_proj(tokens)

        # 4) Self-attention with residual connection and layer norm
        #    mha expects (L, N, E)
        attn_out, _ = self.mha(proj_tokens, proj_tokens, proj_tokens)  # (seq_len, B, model_dim)
        attn_res = self.post_attn_ln(proj_tokens + attn_out)  # residual + LayerNorm

        # 5) Transformer encoder stack
        enc_out = self.encoder(attn_res)  # (seq_len, B, model_dim)

        # 6) Project back to channel feature space and reshape to padded volume
        recon_tokens = self.output_proj(enc_out)  # (seq_len, B, in_channels)
        recon = recon_tokens.permute(1, 2, 0).reshape(B, C, Dp, Hp, Wp)  # (B, C, D', H', W')

        return recon

# Module-level configuration variables
batch_size = 4
in_channels = 8
depth = 4
height = 16
width = 16

model_dim = 32    # must be divisible by nhead
nhead = 4
num_layers = 2
padding = (1, 1, 2, 2, 1, 1)  # (left, right, top, bottom, front, back)
pad_value = 0.1

def get_inputs():
    """
    Returns a list containing a single 5D tensor shaped (batch_size, in_channels, depth, height, width).
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for the Model constructor in the correct order.
    """
    return [in_channels, model_dim, nhead, num_layers, padding, pad_value]