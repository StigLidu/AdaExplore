import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# Configuration (module-level)
batch_size = 4
in_channels = 3
base_channels = 16  # will be expanded inside the model
out_channels = 32
D = 8   # depth
H = 16  # height
W = 16  # width

embed_dim = 256
num_heads = 8
num_layers = 3
negative_slope = 0.02  # for LeakyReLU


class Model(nn.Module):
    """
    Complex module combining 3D convolutions, a projection to a token sequence,
    a Transformer encoder stack, and reconstruction back to volumetric features.
    The pipeline:
      - 3D conv (local feature extractor)
      - flatten spatial dims -> sequence of tokens
      - linear projection to embedding dim + learned positional embedding
      - TransformerEncoder (stack)
      - linear projection back to convolution channels
      - residual add with conv features -> 1x1 Conv3d to out_channels
      - LeakyReLU activation and global average pooling to produce per-batch outputs
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        out_channels: int,
        D: int,
        H: int,
        W: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        negative_slope: float = 0.01,
    ):
        """
        Initializes the model.

        Args:
            in_channels: Number of input channels for the volumetric input.
            base_channels: Base channel count to expand to after the first conv.
            out_channels: Number of output channels after reconstruction conv.
            D, H, W: Spatial dimensions (depth, height, width) of the input volume.
            embed_dim: Token embedding dimension for the Transformer.
            num_heads: Number of attention heads. Must divide embed_dim.
            num_layers: Number of Transformer encoder layers.
            negative_slope: Negative slope parameter for LeakyReLU activation.
        """
        super(Model, self).__init__()

        # Basic checks
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.D = D
        self.H = H
        self.W = W
        self.seq_len = D * H * W
        self.negative_slope = negative_slope

        # Local 3D feature extractor (preserve spatial dims)
        self.conv1 = nn.Conv3d(in_channels, base_channels * 2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(base_channels * 2)

        # Projection from per-voxel feature to transformer embedding
        self.feat_dim = base_channels * 2
        self.proj_to_embed = nn.Linear(self.feat_dim, embed_dim)

        # Learned positional embedding for fixed spatial grid
        self.pos_emb = nn.Parameter(torch.randn(1, self.seq_len, embed_dim))

        # Transformer encoder stack (using batch_first=True so input is (B, seq, E))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Project back from embedding to convolutional feature dimensionality
        self.proj_from_embed = nn.Linear(embed_dim, self.feat_dim)

        # Reconstruction conv and final processing
        self.conv_reconstruct = nn.Conv3d(self.feat_dim, out_channels, kernel_size=1)
        self.bn_reconstruct = nn.BatchNorm3d(out_channels)
        self.activation = nn.LeakyReLU(negative_slope=self.negative_slope, inplace=True)

        # Global pooling to produce a compact per-example output vector
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C_in, D, H, W)

        Returns:
            Tensor of shape (B, out_channels) after global pooling and activation.
        """
        B, C, D_in, H_in, W_in = x.shape
        assert D_in == self.D and H_in == self.H and W_in == self.W, (
            f"Expected spatial dims ({self.D},{self.H},{self.W}), got ({D_in},{H_in},{W_in})"
        )

        # 1) Local 3D conv features
        feat = self.conv1(x)                 # (B, feat_dim, D, H, W)
        feat = self.bn1(feat)
        feat = F.relu(feat)

        # Preserve a residual copy before projection
        residual = feat                      # (B, feat_dim, D, H, W)

        # 2) Flatten spatial dims into sequence of tokens
        # Move channels to last for linear projection: (B, D, H, W, C_feat)
        feat_seq = feat.permute(0, 2, 3, 4, 1).contiguous()
        feat_seq = feat_seq.view(B, self.seq_len, self.feat_dim)  # (B, seq_len, feat_dim)

        # 3) Project to embedding dim and add positional encoding
        tokens = self.proj_to_embed(feat_seq) + self.pos_emb  # (B, seq_len, embed_dim)

        # 4) Transformer encoder processing
        tokens = self.transformer(tokens)  # (B, seq_len, embed_dim)

        # 5) Project back to convolutional feature space
        feat_rec_seq = self.proj_from_embed(tokens)  # (B, seq_len, feat_dim)
        feat_rec = feat_rec_seq.view(B, self.D, self.H, self.W, self.feat_dim)
        feat_rec = feat_rec.permute(0, 4, 1, 2, 3).contiguous()  # (B, feat_dim, D, H, W)

        # 6) Residual fusion and reconstruction conv
        fused = feat_rec + residual  # simple residual linkage
        out_vol = self.conv_reconstruct(fused)  # (B, out_channels, D, H, W)
        out_vol = self.bn_reconstruct(out_vol)
        out_vol = self.activation(out_vol)

        # 7) Global average pooling to produce per-example vector
        pooled = self.global_pool(out_vol)  # (B, out_channels, 1, 1, 1)
        out = pooled.view(B, out_vol.shape[1])  # (B, out_channels)

        return out


def get_inputs() -> List[torch.Tensor]:
    """
    Returns example input tensors for the model.

    - Input volume of shape (batch_size, in_channels, D, H, W)
    """
    x = torch.randn(batch_size, in_channels, D, H, W)
    return [x]


def get_init_inputs() -> List:
    """
    Returns initialization parameters for the Model constructor in the same
    order as the __init__ signature expects them in this example.
    """
    return [
        in_channels,
        base_channels,
        out_channels,
        D,
        H,
        W,
        embed_dim,
        num_heads,
        num_layers,
        negative_slope,
    ]