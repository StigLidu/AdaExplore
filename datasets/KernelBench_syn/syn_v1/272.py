import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex volumetric module that:
    - Applies circular 3D padding
    - Projects volumetric features into a d_model-dimensional embedding via Conv3d
    - Treats spatial locations as a sequence and processes them with a TransformerDecoderLayer
    - Uses a pooled global "memory" expanded across a learned memory-length with positional embeddings
    - Produces a sigmoid gating map to modulate the original input (residual gating)
    """
    def __init__(self, in_channels: int, d_model: int, nhead: int, conv_kernel: int = 3, pad: int = 1, memory_len: int = 8, dim_feedforward: int = 64, dropout: float = 0.1):
        """
        Args:
            in_channels (int): Number of channels in the input volumetric tensor.
            d_model (int): Transformer embedding dimension (also conv output channels).
            nhead (int): Number of attention heads for the TransformerDecoderLayer (must divide d_model).
            conv_kernel (int): Kernel size for the Conv3d projection. conv_kernel should be odd when using circular padding.
            pad (int): CircularPad3d padding applied on each side (int). Use pad = (pad,pad,pad,pad,pad,pad) implicitly.
            memory_len (int): Length of the memory sequence used by the decoder layer.
            dim_feedforward (int): Feedforward dimension in the TransformerDecoderLayer.
            dropout (float): Dropout probability in the TransformerDecoderLayer.
        """
        super(Model, self).__init__()

        # Basic building blocks
        self.pad = pad
        # Apply circular padding to preserve spatial dims before conv with kernel>1
        self.circular_pad = nn.CircularPad3d(pad)

        # Project input channels to d_model feature channels via 3D conv.
        # Use padding=0 because we already applied circular padding.
        self.conv_proj = nn.Conv3d(in_channels=in_channels, out_channels=d_model, kernel_size=conv_kernel, stride=1, padding=0, bias=True)

        # Transformer decoder layer: acts on target sequence (spatial locations) with global memory
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation='relu')

        # Projection of pooled global features into memory embeddings
        self.memory_proj = nn.Linear(d_model, d_model)

        # Learned positional embeddings for memory sequence (memory_len x d_model)
        self.memory_pos = nn.Parameter(torch.randn(memory_len, d_model))

        # Project transformer outputs back to input channels
        self.out_proj_conv = nn.Conv3d(in_channels=d_model, out_channels=in_channels, kernel_size=1, stride=1, padding=0, bias=True)

        # Sigmoid gating to modulate input with the transformed features
        self.sigmoid = nn.Sigmoid()

        # Store config
        self.d_model = d_model
        self.memory_len = memory_len
        self.conv_kernel = conv_kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input volumetric tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of same shape as input, where a learned gated residual is applied.
        """
        # 1) Circular padding
        x_padded = self.circular_pad(x)  # shape: (N, C, D+2*pad, H+2*pad, W+2*pad)

        # 2) Project to d_model channels via Conv3d
        feat = self.conv_proj(x_padded)  # shape: (N, d_model, D, H, W) assuming padding preserves dims

        # 3) Prepare sequence from spatial locations: (seq_len, batch, d_model)
        N, C, D, H, W = feat.shape
        seq_len = D * H * W
        # flatten spatial dims -> (N, d_model, seq_len)
        feat_seq = feat.view(N, C, seq_len).permute(2, 0, 1).contiguous()  # (seq_len, N, d_model)

        # 4) Build a compact global memory from pooled features and expand to memory_len
        pooled = feat.mean(dim=(2, 3, 4))  # (N, d_model)
        mem_emb = self.memory_proj(pooled)  # (N, d_model)
        # expand to (memory_len, N, d_model) and add learned positional embeddings (memory_len, d_model)
        memory = mem_emb.unsqueeze(0).repeat(self.memory_len, 1, 1) + self.memory_pos.unsqueeze(1)  # (memory_len, N, d_model)

        # 5) Run one Transformer decoder layer: decoder_layer expects (tgt, memory)
        decoded = self.decoder_layer(tgt=feat_seq, memory=memory)  # (seq_len, N, d_model)

        # 6) Map back to volumetric shape: (N, d_model, D, H, W)
        decoded_vol = decoded.permute(1, 2, 0).contiguous().view(N, C, D, H, W)

        # 7) Project to input channels and compute gating via sigmoid
        gate = self.sigmoid(self.out_proj_conv(decoded_vol))  # (N, in_channels, D, H, W)

        # 8) Apply gated residual: output = input * gate + input * (1 - gate) * 0.5 (blend with scaled residual)
        out = x * gate + x * (1.0 - gate) * 0.5

        return out

# Module-level configuration variables
batch_size = 4
in_channels = 8
depth = 8
height = 8
width = 8

d_model = 16  # must be divisible by nhead
nhead = 4
conv_kernel = 3
pad = 1
memory_len = 6
dim_feedforward = 64
dropout = 0.1

def get_inputs():
    # Random volumetric input
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    # Return initialization parameters for Model.__init__
    return [in_channels, d_model, nhead, conv_kernel, pad, memory_len, dim_feedforward, dropout]