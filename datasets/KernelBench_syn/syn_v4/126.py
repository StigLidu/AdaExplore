import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex model that combines 2D convolutions, a lazy BatchNorm, and multi-head attention
    to process image-like inputs and produce classification logits.

    Computation steps (high-level):
    1. conv1 (3x3) + ReLU -> intermediate feature maps
    2. conv2 (1x1) -> per-location embeddings
    3. Flatten spatial dims to sequence and apply LazyBatchNorm1d over feature dim
       (batch*seq treated as the batch for normalization)
    4. MultiheadAttention over the spatial sequence (self-attention)
    5. Residual connection from step 2 embeddings to attention output
    6. conv3 (3x3) as a local feed-forward applied on the attention-refined features
    7. Global average pooling and final linear classifier
    """
    def __init__(self,
                 in_channels: int = 3,
                 mid_channels: int = 64,
                 embed_dim: int = 128,
                 num_heads: int = 8,
                 num_classes: int = 10):
        super(Model, self).__init__()
        # Initial local feature extractor
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=True)
        # Project to embedding dimension per spatial location (1x1 conv acts like pointwise linear)
        self.conv2 = nn.Conv2d(mid_channels, embed_dim, kernel_size=1, bias=True)
        # Lazy BatchNorm1d will infer num_features on first forward pass
        self.bn = nn.LazyBatchNorm1d()  # normalizes over feature dimension after flattening (B * S, E)
        # Multi-head self-attention operating on spatial sequence (seq_len, batch, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=False)
        # Local feed-forward (refinement) applied spatially
        self.conv3 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=True)
        # Final classifier
        self.classifier = nn.Linear(embed_dim, num_classes)
        # A small dropout for regularization (optional)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Logits of shape (B, num_classes)
        """
        B, C, H, W = x.shape
        # 1) Local conv + non-linearity
        x_local = self.conv1(x)            # (B, mid_channels, H, W)
        x_local = F.relu(x_local, inplace=True)
        # 2) Project to per-location embeddings
        x_embed = self.conv2(x_local)      # (B, embed_dim, H, W)

        # 3) Flatten spatial dims into a sequence for attention
        seq_len = H * W
        # (B, E, H*W)
        x_flat = x_embed.view(B, x_embed.size(1), seq_len)
        # Permute to (seq_len, B, E) for attention module
        x_seq = x_flat.permute(2, 0, 1).contiguous()  # (S, B, E)

        # 4) Apply LazyBatchNorm1d across feature dimension treating (B * S) as batch dimension
        # Move to shape (B*S, E) for batchnorm
        x_bn_in = x_seq.permute(1, 0, 2).contiguous().view(B * seq_len, -1)  # (B*S, E)
        x_bn_out = self.bn(x_bn_in)  # LazyBatchNorm1d will initialize on first call
        # Reshape back to (S, B, E)
        x_bn = x_bn_out.view(B, seq_len, -1).permute(1, 0, 2).contiguous()  # (S, B, E)

        # 5) Self-attention (queries, keys, values are the same for self-attention)
        attn_out, attn_weights = self.attn(x_bn, x_bn, x_bn)  # (S, B, E), (B * num_heads, S, S) or similar
        # 6) Residual connection
        x_res = attn_out + x_seq  # (S, B, E)

        # 7) Convert back to spatial feature map (B, E, H, W)
        x_spatial = x_res.permute(1, 2, 0).contiguous().view(B, -1, H, W)  # (B, E, H, W)

        # 8) Local refinement conv + non-linearity
        x_refined = self.conv3(x_spatial)  # (B, E, H, W)
        x_refined = F.relu(x_refined, inplace=True)
        x_refined = self.dropout(x_refined)

        # 9) Global average pooling over spatial dims -> (B, E)
        out_pool = x_refined.view(B, x_refined.size(1), -1).mean(dim=2)

        # 10) Final classification linear layer -> (B, num_classes)
        logits = self.classifier(out_pool)

        return logits

# Configuration / default sizes for get_inputs
BATCH = 8
IN_CHANNELS = 3
HEIGHT = 64
WIDTH = 64
MID_CHANNELS = 64
EMBED_DIM = 128
NUM_HEADS = 8
NUM_CLASSES = 10

def get_inputs():
    """
    Create a single random image batch input for testing.

    Returns:
        list: [x] where x has shape (BATCH, IN_CHANNELS, HEIGHT, WIDTH)
    """
    x = torch.randn(BATCH, IN_CHANNELS, HEIGHT, WIDTH)
    return [x]

def get_init_inputs():
    """
    Provide constructor arguments for the Model.

    Returns:
        list: [in_channels, mid_channels, embed_dim, num_heads, num_classes]
    """
    return [IN_CHANNELS, MID_CHANNELS, EMBED_DIM, NUM_HEADS, NUM_CLASSES]