import torch
import torch.nn as nn
from typing import List, Tuple

class Model(nn.Module):
    """
    Complex 3D-to-sequence model that:
    - Applies 3D max pooling to reduce spatial resolution.
    - Projects channel features into a transformer embedding space with a 1x1x1 Conv3d.
    - Treats the pooled spatial grid as a sequence and processes it with a TransformerEncoder.
    - Applies a Tanhshrink non-linearity, aggregates across the sequence, and projects to an output space via an external matrix.
    """
    def __init__(
        self,
        in_channels: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        pool_kernel: Tuple[int, int, int],
        out_features: int,
    ):
        """
        Initializes layers used in the forward pass.

        Args:
            in_channels (int): Number of input channels in the 3D volume.
            d_model (int): Transformer embedding dimension.
            nhead (int): Number of attention heads for the transformer.
            num_layers (int): Number of TransformerEncoder layers.
            pool_kernel (tuple): Kernel (and stride) size for MaxPool3d.
            out_features (int): Output feature dimension after final projection.
        """
        super(Model, self).__init__()
        self.pool = nn.MaxPool3d(kernel_size=pool_kernel, stride=pool_kernel)
        # 1x1x1 conv to map channels -> d_model (acts like a per-location linear projection)
        self.channel_proj = nn.Conv3d(in_channels, d_model, kernel_size=1, bias=True)
        # Transformer encoder stack (batch_first for convenience)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Non-linearity applied element-wise after transformer
        self.tanhs = nn.Tanhshrink()
        # Small final linear to optionally refine outputs (not strictly necessary since we accept external matrix)
        self.refine = nn.Linear(d_model, d_model)
        # Keep out_features for documentation; final projection matrix is expected as an input to forward
        self.out_features = out_features
        # Layer normalization for stable training/inference
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, projection_matrix: torch.Tensor) -> torch.Tensor:
        """
        Forward computation.

        Args:
            x (torch.Tensor): Input 3D volume tensor of shape (B, C, D, H, W).
            projection_matrix (torch.Tensor): External projection matrix of shape (d_model, out_features).

        Returns:
            torch.Tensor: Output tensor of shape (B, out_features).
        """
        # 1) Spatial downsampling
        x = self.pool(x)  # (B, C, pD, pH, pW)

        # 2) Channel projection to transformer embedding dimension
        x = self.channel_proj(x)  # (B, d_model, pD, pH, pW)

        # 3) Flatten spatial grid to a sequence (batch_first for transformer)
        B, E, pD, pH, pW = x.shape
        seq_len = pD * pH * pW
        x = x.view(B, E, seq_len).permute(0, 2, 1).contiguous()  # (B, seq_len, d_model)

        # 4) Transformer encoding
        x = self.encoder(x)  # (B, seq_len, d_model)

        # 5) Non-linearity and refinement
        x = self.tanhs(x)  # (B, seq_len, d_model)
        x = self.refine(x)  # (B, seq_len, d_model)
        x = self.norm(x)  # (B, seq_len, d_model)

        # 6) Aggregate across sequence and apply external projection
        x = x.mean(dim=1)  # Global mean: (B, d_model)
        # projection_matrix should be (d_model, out_features)
        out = x.matmul(projection_matrix)  # (B, out_features)
        return out

# Configuration variables
batch_size = 8
in_channels = 12
D = 8
H = 16
W = 16

d_model = 48        # Must be divisible by nhead
nhead = 8
num_layers = 3
pool_kernel = (2, 2, 2)
out_features = 128

def get_inputs() -> List[torch.Tensor]:
    """
    Returns a list of input tensors to pass to the model's forward method:
    - 3D volume tensor of shape (batch_size, in_channels, D, H, W)
    - External projection matrix of shape (d_model, out_features)
    """
    x = torch.randn(batch_size, in_channels, D, H, W)
    projection_matrix = torch.randn(d_model, out_features)
    return [x, projection_matrix]

def get_init_inputs() -> List:
    """
    Returns the initialization parameters expected by Model.__init__ in order:
    [in_channels, d_model, nhead, num_layers, pool_kernel, out_features]
    """
    return [in_channels, d_model, nhead, num_layers, pool_kernel, out_features]