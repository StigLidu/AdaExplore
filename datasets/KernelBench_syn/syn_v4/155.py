import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex model that combines 2D convolutions with a Transformer encoder over image patches.
    Pipeline:
      - Initial Conv2d to extract feature maps from an RGB image.
      - Patchify the feature map into non-overlapping patches, project each patch to a token embedding.
      - Process the token sequence with an nn.TransformerEncoder (stack of encoder layers).
      - Reconstruct spatial feature map from transformer tokens.
      - Concatenate reconstructed map with early convolution features and apply a LazyConv2d
        to produce the final output map. LazyConv2d determines its in_channels at the first call.
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        patch_size: int,
        mid_channels: int,
        recon_channels: int,
        final_out_channels: int,
    ):
        """
        Args:
            d_model: Transformer embedding dimension.
            nhead: Number of attention heads in the Transformer.
            num_layers: Number of Transformer encoder layers.
            patch_size: Height and width of non-overlapping patches (patch_size x patch_size).
            mid_channels: Number of channels produced by the initial Conv2d feature extractor.
            recon_channels: Number of channels used to reconstruct spatial patches before final conv.
            final_out_channels: Number of output channels produced by the final LazyConv2d.
        """
        super(Model, self).__init__()
        self.patch_size = patch_size
        self.mid_channels = mid_channels
        self.recon_channels = recon_channels

        # Initial feature extractor: produces a dense feature map from the input image
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=mid_channels, kernel_size=3, stride=1, padding=1)

        # The dimension of each patch vector (before projection) depends on mid_channels and patch_size
        patch_area = patch_size * patch_size
        self.patch_vec_dim = mid_channels * patch_area

        # Linear projection from patch vector to transformer embedding dimension
        self.patch_proj = nn.Linear(self.patch_vec_dim, d_model)

        # Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=0.1, activation='relu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Project transformer outputs back to patch vectors for reconstruction
        self.out_proj = nn.Linear(d_model, recon_channels * patch_area)

        # Final convolution that mixes the concatenated features; in_channels will be lazily set
        self.conv2 = nn.LazyConv2d(out_channels=final_out_channels, kernel_size=3, stride=1, padding=1)

        # Small activation
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input image tensor of shape (B, 3, H, W). H and W must be divisible by patch_size.

        Returns:
            Tensor of shape (B, final_out_channels, H, W).
        """
        B, C, H, W = x.shape
        ph = pw = self.patch_size
        assert H % ph == 0 and W % pw == 0, "Height and width must be divisible by patch_size"

        # 1) Initial convolution and activation
        feat = self.act(self.conv1(x))  # (B, mid_channels, H, W)

        # 2) Patchify feature map into (B, L, patch_vec_dim)
        new_H = H // ph
        new_W = W // pw
        # reshape to (B, mid_channels, new_H, ph, new_W, pw)
        feat_reshaped = feat.view(B, self.mid_channels, new_H, ph, new_W, pw)
        # permute to (B, new_H, new_W, mid_channels, ph, pw)
        feat_reshaped = feat_reshaped.permute(0, 2, 4, 1, 3, 5).contiguous()
        L = new_H * new_W
        # flatten patch content to vectors: (B, L, patch_vec_dim)
        patches = feat_reshaped.view(B, L, self.patch_vec_dim)

        # 3) Project patches to transformer embeddings and run through TransformerEncoder
        tokens = self.patch_proj(patches)               # (B, L, d_model)
        tokens = tokens.permute(1, 0, 2).contiguous()   # (L, B, d_model) expected by Transformer
        tokens = self.encoder(tokens)                   # (L, B, d_model)
        tokens = tokens.permute(1, 0, 2).contiguous()   # (B, L, d_model)

        # 4) Project transformer outputs back to patch vectors and reconstruct spatial map
        recon = self.out_proj(tokens)                   # (B, L, recon_channels * patch_area)
        recon = recon.view(B, new_H, new_W, self.recon_channels, ph, pw)
        recon = recon.permute(0, 3, 1, 4, 2, 5).contiguous()  # (B, recon_channels, new_H, ph, new_W, pw)
        recon = recon.view(B, self.recon_channels, H, W)     # (B, recon_channels, H, W)

        # 5) Concatenate reconstructed map with early features and apply final LazyConv2d
        cat = torch.cat([recon, feat], dim=1)  # (B, recon_channels + mid_channels, H, W)
        out = self.act(self.conv2(cat))        # LazyConv2d will set in_channels on first call

        return out

# Configuration variables (example sizes)
batch_size = 4
channels = 3
height = 64
width = 64

# Transformer and model hyperparameters
d_model = 256
nhead = 8
num_layers = 3
patch_size = 8
mid_channels = 32
recon_channels = 48
final_out_channels = 16

def get_inputs():
    # Random image batch; H and W are chosen to be divisible by patch_size
    x = torch.randn(batch_size, channels, height, width)
    return [x]

def get_init_inputs():
    # Return initialization arguments for Model in the correct order
    return [d_model, nhead, num_layers, patch_size, mid_channels, recon_channels, final_out_channels]