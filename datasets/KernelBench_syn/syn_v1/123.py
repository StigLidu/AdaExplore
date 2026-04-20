import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A moderately complex module that:
    - Applies adaptive max pooling to spatial input (B, C, H, W) -> (B, C, pH, pW)
    - Flattens pooled features and projects to a lower-dim embedding
    - Applies Softplus non-linearity and L2-normalizes embeddings
    - Computes a batch-wise similarity matrix and converts it to attention weights via Softmax
    - Uses attention to aggregate original pooled features across the batch (contextualization)
    - Adds a residual connection and produces a final projected output with another Softplus

    This model demonstrates use of nn.AdaptiveMaxPool2d, nn.Softplus, and nn.Softmax,
    combined with linear projections and matrix operations to form a novel computation pattern.
    """
    def __init__(self,
                 in_channels: int,
                 pool_output: tuple,
                 proj_dim: int,
                 out_dim: int,
                 eps: float = 1e-6):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.pool_output = pool_output  # (pH, pW)
        self.proj_dim = proj_dim
        self.out_dim = out_dim
        self.eps = eps

        # Layers
        self.pool = nn.AdaptiveMaxPool2d(self.pool_output)
        flattened_size = in_channels * self.pool_output[0] * self.pool_output[1]
        self.encoder = nn.Linear(flattened_size, self.proj_dim, bias=True)
        self.softplus = nn.Softplus()
        # Softmax over dimension 1 (rows of similarity -> attention distribution per row)
        self.softmax = nn.Softmax(dim=1)
        # Project back to a desired output dimension
        self.out_proj = nn.Linear(flattened_size, self.out_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            out: Tensor of shape (B, out_dim)
        """
        # Step 1: Adaptive max pooling to fixed spatial size
        # -> (B, C, pH, pW)
        pooled = self.pool(x)

        B = pooled.shape[0]
        # Step 2: Flatten spatial and channel dims -> (B, F)
        F = self.in_channels * self.pool_output[0] * self.pool_output[1]
        pooled_flat = pooled.view(B, F)

        # Step 3: Encode to lower-dim embedding and apply Softplus
        emb = self.encoder(pooled_flat)  # (B, proj_dim)
        emb = self.softplus(emb)         # (B, proj_dim)

        # Step 4: L2-normalize embeddings to prepare for cosine-like similarity
        norm = emb.norm(p=2, dim=1, keepdim=True).clamp_min(self.eps)
        emb_norm = emb / norm             # (B, proj_dim)

        # Step 5: Compute batch-wise similarity matrix (B x B)
        sim = torch.matmul(emb_norm, emb_norm.t())  # (B, B)

        # Step 6: Convert similarities into attention weights (rows sum to 1)
        attn = self.softmax(sim)  # (B, B)

        # Step 7: Use attention to aggregate original pooled features across the batch
        # This produces a context vector per sample: weighted sum of pooled_flat over batch
        context = torch.matmul(attn, pooled_flat)  # (B, F)

        # Step 8: Residual connection combining local pooled features with context
        combined = pooled_flat + context  # (B, F)

        # Step 9: Final projection and non-linearity -> (B, out_dim)
        out = self.out_proj(combined)
        out = self.softplus(out)

        return out


# Configuration variables
batch_size = 32
in_channels = 16
input_h = 64
input_w = 64
pool_output = (4, 4)   # Adaptive pooling target spatial dims
proj_dim = 128
out_dim = 512

def get_inputs():
    """
    Returns:
        A single-element list with the input tensor of shape (batch_size, in_channels, H, W).
    """
    x = torch.randn(batch_size, in_channels, input_h, input_w, dtype=torch.float32)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters used to construct the Model instance.
    This is useful for frameworks that recreate the module prior to benchmarking.
    """
    return [in_channels, pool_output, proj_dim, out_dim]