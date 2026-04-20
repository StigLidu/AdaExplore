import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    A composite model that:
    - Uses nn.EmbeddingBag to aggregate variable-length token bags into bag embeddings.
    - Projects embeddings with a Linear layer.
    - Concatenates with additional dense features.
    - Expands the combined features into a 4D tensor and applies InstanceNorm2d.
    - Applies a non-linearity and reduces back to a vector via adaptive pooling and a final Linear layer.
    This pattern demonstrates sparse bag aggregation, dense projection, and normalization over spatial dims.
    """
    def __init__(
        self,
        num_embeddings: int,
        emb_dim: int,
        proj_dim: int,
        dense_dim: int,
        channels: int,
        height: int,
        width: int,
        out_dim: int,
        emb_mode: str = "mean",
        eps: float = 1e-5,
        affine: bool = True,
    ):
        """
        Args:
            num_embeddings: Size of the embedding dictionary.
            emb_dim: Dimensionality of each embedding vector.
            proj_dim: Dimensionality to project the embedding into.
            dense_dim: Dimensionality of the auxiliary dense features provided per bag.
            channels: Number of channels to expand into for InstanceNorm2d.
            height: Spatial height after expansion.
            width: Spatial width after expansion.
            out_dim: Output vector dimensionality.
            emb_mode: Mode for EmbeddingBag aggregation ('sum' or 'mean').
            eps: Epsilon for InstanceNorm2d.
            affine: Whether InstanceNorm2d has learnable affine parameters.
        """
        super(Model, self).__init__()
        # Embedding bag for variable-length bag aggregation
        self.emb_bag = nn.EmbeddingBag(num_embeddings, emb_dim, mode=emb_mode, sparse=False)

        # Project the embedding to a smaller/larger hidden dimension
        self.proj_emb = nn.Linear(emb_dim, proj_dim, bias=True)

        # After concatenation with dense features, map to channels * H * W
        self.concat_to_spatial = nn.Linear(proj_dim + dense_dim, channels * height * width, bias=True)

        # Instance normalization on the expanded spatial tensor
        self.inst_norm = nn.InstanceNorm2d(channels, eps=eps, affine=affine)

        # Final classifier/regressor head from pooled features
        self.final_head = nn.Linear(channels, out_dim, bias=True)

        # Store shapes
        self.channels = channels
        self.height = height
        self.width = width

    def forward(self, indices: torch.LongTensor, offsets: torch.LongTensor, dense: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            indices: 1D tensor of token indices for all bags concatenated (shape: total_tokens).
            offsets: 1D tensor of bag offsets (shape: batch_size).
            dense: Dense feature tensor per bag (shape: batch_size, dense_dim).

        Returns:
            Tensor of shape (batch_size, out_dim)
        """
        # 1) Aggregate token embeddings per bag -> (B, emb_dim)
        bag_emb = self.emb_bag(indices, offsets)  # EmbeddingBag aggregation

        # 2) Project embedding -> (B, proj_dim)
        p = self.proj_emb(bag_emb)

        # 3) Concatenate with dense features -> (B, proj_dim + dense_dim)
        combined = torch.cat([p, dense], dim=-1)

        # 4) Map to spatial tensor size and reshape -> (B, C, H, W)
        spatial = self.concat_to_spatial(combined)
        B = spatial.shape[0]
        spatial = spatial.view(B, self.channels, self.height, self.width)

        # 5) Apply InstanceNorm2d -> (B, C, H, W)
        spatial = self.inst_norm(spatial)

        # 6) Non-linearity
        spatial = F.relu(spatial, inplace=False)

        # 7) Global average over spatial dims -> (B, C)
        pooled = spatial.mean(dim=[2, 3])

        # 8) Final linear head -> (B, out_dim)
        out = self.final_head(pooled)
        return out

# Configuration / hyperparameters
batch_size = 8
num_embeddings = 1024
emb_dim = 128
proj_dim = 64
dense_dim = 32
channels = 16
height = 8
width = 8
out_dim = 10
max_tokens_per_bag = 50
min_tokens_per_bag = 1
emb_mode = "mean"

def get_inputs():
    """
    Constructs realistic inputs for EmbeddingBag: a concatenated indices tensor and offsets per bag,
    plus an auxiliary dense feature tensor.
    """
    # Random variable-length bag sizes for each sample in the batch
    lengths = torch.randint(low=min_tokens_per_bag, high=max_tokens_per_bag + 1, size=(batch_size,))
    total_tokens = int(lengths.sum().item())

    # Random indices for tokens across the vocabulary
    indices = torch.randint(low=0, high=num_embeddings, size=(total_tokens,), dtype=torch.long)

    # Offsets: starting index of each bag in the concatenated indices tensor
    offsets = torch.empty(batch_size, dtype=torch.long)
    offsets[0] = 0
    torch.cumsum(lengths, dim=0, out=offsets)
    offsets = offsets - lengths  # compute starting positions

    # Dense auxiliary features per bag
    dense = torch.randn(batch_size, dense_dim)

    return [indices, offsets, dense]

def get_init_inputs():
    """
    Returns the initialization parameters for the Model constructor in the same order.
    """
    return [num_embeddings, emb_dim, proj_dim, dense_dim, channels, height, width, out_dim, emb_mode]