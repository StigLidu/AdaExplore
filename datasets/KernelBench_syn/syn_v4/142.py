import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex model that:
    - Pools spatial features with AdaptiveAvgPool2d to a small spatial grid.
    - Projects spatial locations to a token sequence for a TransformerDecoder.
    - Builds a compact "memory" token from a global pooling of the original image.
    - Runs a TransformerDecoder stack to refine spatial tokens conditioned on the memory.
    - Reconstructs a feature map from decoder outputs, upsamples back to input resolution,
      and applies a channel-wise PReLU activation.
    """
    def __init__(
        self,
        in_channels: int,
        pool_h: int,
        pool_w: int,
        embed_dim: int,
        num_layers: int,
        nhead: int,
        dim_feedforward: int,
        upsample_mode: str = "bilinear",
    ):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.upsample_mode = upsample_mode

        # Pool to a fixed (pool_h, pool_w) grid to create spatial tokens
        self.spatial_pool = nn.AdaptiveAvgPool2d((pool_h, pool_w))
        # Global pool to create a compact memory vector (1 token)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Linear projection from input channels -> transformer embedding for tokens and memory
        self.token_proj = nn.Linear(in_channels, embed_dim)   # applied on last dim of tokens
        self.memory_proj = nn.Linear(in_channels, embed_dim)  # applied on global pooled vector

        # Transformer decoder stack: refines tokens conditioned on memory token(s)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Reconstruct channels from embed_dim after decoder and upsample back to input size
        self.reconstruct = nn.Conv2d(embed_dim, in_channels, kernel_size=1)

        # Channel-wise adaptive activation
        self.prelu = nn.PReLU(num_parameters=in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W) after transformer-conditioned refinement
        """
        B, C, H, W = x.shape

        # 1) Spatial pooling to create tokens: (B, C, pool_h, pool_w)
        pooled = self.spatial_pool(x)

        # 2) Convert spatial grid to sequence of tokens:
        #    pooled -> (B, C, pool_h * pool_w) -> (pool_h*pool_w, B, C)
        S = self.pool_h * self.pool_w
        tokens = pooled.flatten(2).permute(2, 0, 1)  # (S, B, C)

        # 3) Project tokens to embed_dim (Transformer expects shape (S, B, E))
        tokens = self.token_proj(tokens)  # (S, B, embed_dim)

        # 4) Build memory from global pooled features (1 token per batch)
        global_vec = self.global_pool(x).view(B, C)  # (B, C)
        memory = self.memory_proj(global_vec).unsqueeze(0)  # (1, B, embed_dim)

        # 5) Run TransformerDecoder: target=tokens, memory=memory -> (S, B, embed_dim)
        decoded = self.decoder(tgt=tokens, memory=memory)

        # 6) Reshape decoded tokens back to spatial map: (B, embed_dim, pool_h, pool_w)
        decoded_spatial = decoded.permute(1, 2, 0).contiguous().view(B, self.embed_dim, self.pool_h, self.pool_w)

        # 7) Reconstruct to input channels and upsample to original resolution
        reconstructed = self.reconstruct(decoded_spatial)  # (B, C, pool_h, pool_w)
        if self.upsample_mode == "bilinear":
            out = F.interpolate(reconstructed, size=(H, W), mode="bilinear", align_corners=False)
        else:
            out = F.interpolate(reconstructed, size=(H, W), mode=self.upsample_mode)

        # 8) Final channel-wise PReLU activation
        return self.prelu(out)

# Configuration / initialization parameters
batch_size = 8
in_channels = 32
height = 128
width = 128

pool_h = 8
pool_w = 8

embed_dim = 64
num_layers = 3
nhead = 8
dim_feedforward = 256
upsample_mode = "bilinear"

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    # Returns the parameters required to initialize the Model in the same order as __init__
    return [in_channels, pool_h, pool_w, embed_dim, num_layers, nhead, dim_feedforward, upsample_mode]