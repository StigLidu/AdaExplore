import torch
import torch.nn as nn

"""
Complex PyTorch module that combines 3D pooling/unpooling, a lazy 2D convolution applied slice-wise,
and a Transformer decoder layer to perform cross-slice contextual refinement.

Computation overview:
1. MaxPool3d over spatial dims (H, W) with return_indices to keep positions for unpooling.
2. Reshape pooled volume to treat each depth slice as an independent 2D image and apply LazyConv2d.
3. Global-average each slice to form a sequence of embeddings (depth dimension is the sequence).
4. Use a TransformerDecoderLayer to let slices attend to each other (contextual refinement).
5. Decode transformer outputs back to spatial maps via a Linear layer, reshape to match pooled spatial
   resolution and use MaxUnpool3d with stored indices to reconstruct original spatial resolution.
6. Add a residual connection to the original input.

Module-level configuration variables control dimensions and pooling behavior.
"""

# Configuration / shapes
batch_size = 4
in_channels = 1
depth = 8
height = 64
width = 64

# Pooling parameters (we only pool spatial HxW, keep depth unpooled)
pool_kernel = (1, 2, 2)
pool_stride = (1, 2, 2)

# Derived pooled spatial sizes (assume divisible for simplicity)
pooled_height = height // pool_kernel[1]
pooled_width = width // pool_kernel[2]

# Feature sizes
conv_out_channels = 16  # embedding dimension for transformer must match this
transformer_embed_dim = conv_out_channels
transformer_nhead = 4  # must divide embed_dim; 16 % 4 == 0

class Model(nn.Module):
    """
    Model that integrates MaxPool3d/MaxUnpool3d, LazyConv2d, and TransformerDecoderLayer.
    Processes a volumetric input (B, C=1, D, H, W), applies slice-wise conv, cross-slice
    transformer refinement, and reconstructs a refined volume.
    """
    def __init__(self):
        super(Model, self).__init__()
        # 3D pooling/unpooling over spatial dims (keep depth intact)
        self.pool = nn.MaxPool3d(kernel_size=pool_kernel, stride=pool_stride, return_indices=True)
        self.unpool = nn.MaxUnpool3d(kernel_size=pool_kernel, stride=pool_stride)

        # LazyConv2d will infer in_channels from first forward call (we feed single-channel slices)
        self.slice_conv = nn.LazyConv2d(out_channels=conv_out_channels, kernel_size=3, padding=1)

        # Transformer decoder layer for cross-slice contextualization
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=transformer_embed_dim, nhead=transformer_nhead)

        # Linear to decode transformer embeddings back to spatial map (pooled HxW)
        self.embedding_to_spatial = nn.Linear(transformer_embed_dim, pooled_height * pooled_width)

        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C=1, D, H, W)

        Returns:
            Refined volume of same shape (B, C=1, D, H, W)
        """
        # Step 1: Pool spatial dimensions, keep depth dimension intact
        # p: (B, C, D, H2, W2), indices same shape
        p, indices = self.pool(x)

        B, C, D, H2, W2 = p.shape  # C expected to be 1 per config

        # Step 2: Treat each depth slice as an independent 2D image and apply conv2d slice-wise
        # Reshape to (B*D, C, H2, W2)
        p_slices = p.permute(0, 2, 1, 3, 4).contiguous().view(B * D, C, H2, W2)

        # LazyConv2d will initialize on first call
        conv_features = self.activation(self.slice_conv(p_slices))  # (B*D, conv_out_channels, H2, W2)

        # Step 3: Global average pool spatial dims to get per-slice embeddings
        # (B*D, conv_out_channels) -> (B, D, conv_out_channels)
        embeddings = conv_features.mean(dim=[2, 3]).view(B, D, conv_out_channels)

        # Prepare sequences for transformer: (seq_len=D, batch=B, embed_dim)
        tgt = embeddings.permute(1, 0, 2)   # (D, B, embed)
        memory = tgt  # using same sequence as memory for self-contextualization

        # Step 4: TransformerDecoderLayer to allow slices to attend to each other
        decoded = self.decoder_layer(tgt, memory)  # (D, B, embed_dim)

        # Step 5: Decode transformer outputs back to spatial map and reshape for unpooling
        decoded_bxd = decoded.permute(1, 0, 2).contiguous().view(B * D, transformer_embed_dim)  # (B*D, embed)
        spatial_flat = self.embedding_to_spatial(decoded_bxd)  # (B*D, H2*W2)
        spatial_map = spatial_flat.view(B * D, 1, H2, W2)  # reduce to 1 channel to match pooling indices

        # Reshape to (B, 1, D, H2, W2) to unpool
        to_unpool = spatial_map.view(B, 1, D, H2, W2)

        # Step 6: Unpool back to original spatial resolution using stored indices
        out = self.unpool(to_unpool, indices, output_size=x.shape)

        # Residual connection to original input
        return out + x

def get_inputs():
    """
    Generate a sample volumetric input tensor for testing.

    Returns:
        list: [x] where x has shape (batch_size, in_channels, depth, height, width)
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    No special initialization parameters required; LazyConv2d will initialize on first forward.

    Returns:
        list: empty
    """
    return []