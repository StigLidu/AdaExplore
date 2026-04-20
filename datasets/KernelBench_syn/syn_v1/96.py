import torch
import torch.nn as nn

"""
Complex example combining EmbeddingBag, Softplus, and AdaptiveMaxPool1d.

Computation pipeline:
1. Use nn.EmbeddingBag to convert variable-length token bags into dense embeddings per sample.
2. Project embeddings with a Linear layer and apply Softplus non-linearity.
3. Expand the per-sample vectors into (channels, seq_len) tensors via a second Linear layer,
   then reshape to (batch, channels, seq_len).
4. Apply nn.AdaptiveMaxPool1d to reduce the temporal dimension.
5. Flatten and apply a final Linear projection to produce the outputs.
"""

# Configuration
batch_size = 12
num_embeddings = 10000
embedding_dim = 128
proj_dim = 64          # number of channels after projection
seq_len = 16           # temporal length created from each projected vector
pool_out_len = 4       # output temporal length after adaptive pooling
final_dim = 32         # final output dimension
max_bag_len = 20       # maximum number of indices (tokens) per bag/sample


class Model(nn.Module):
    """
    Model that processes bags of token indices using EmbeddingBag, followed by projection,
    Softplus activation, unfolding into a (C, L) tensor per sample, adaptive max pooling,
    and a final projection.

    Forward signature:
        forward(indices: torch.LongTensor, offsets: torch.LongTensor) -> torch.Tensor

    - indices: 1D tensor with concatenated token indices for all samples (dtype=torch.long)
    - offsets: 1D tensor with starting index of each sample in `indices` (dtype=torch.long)
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        proj_dim: int,
        seq_len: int,
        pool_out_len: int,
        final_dim: int,
    ):
        super(Model, self).__init__()
        # EmbeddingBag aggregates variable-length bags into fixed-size embeddings per sample
        self.embedding_bag = nn.EmbeddingBag(num_embeddings=num_embeddings, embedding_dim=embedding_dim, mode='mean', sparse=False)

        # Project embedding_dim -> proj_dim
        self.proj = nn.Linear(embedding_dim, proj_dim)

        # Non-linearity
        self.softplus = nn.Softplus()

        # Expand each projected vector into a flattened (proj_dim * seq_len) vector,
        # then reshape into (proj_dim, seq_len) per sample
        self.unfold = nn.Linear(proj_dim, proj_dim * seq_len)

        # Adaptive 1D max pooling across the temporal dimension
        self.pool = nn.AdaptiveMaxPool1d(output_size=pool_out_len)

        # Final projection from flattened pooled representation to desired final_dim
        pooled_flat_dim = proj_dim * pool_out_len
        self.fc = nn.Linear(pooled_flat_dim, final_dim)

    def forward(self, indices: torch.LongTensor, offsets: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            indices: 1D tensor of token indices, shape (total_tokens,)
            offsets: 1D tensor of starting offsets for each bag/sample, shape (batch_size,)

        Returns:
            Tensor of shape (batch_size, final_dim)
        """
        # 1) Aggregate embeddings per bag/sample
        emb = self.embedding_bag(indices, offsets)  # shape: (batch_size, embedding_dim)

        # 2) Project and apply Softplus
        projected = self.proj(emb)                  # (batch_size, proj_dim)
        activated = self.softplus(projected)        # (batch_size, proj_dim)

        # 3) Expand to (batch_size, proj_dim * seq_len) then reshape to (batch, C, L)
        expanded = self.unfold(activated)           # (batch_size, proj_dim * seq_len)
        batch = expanded.shape[0]
        x = expanded.view(batch, -1, seq_len)       # (batch_size, proj_dim, seq_len)

        # 4) Adaptive max pooling -> (batch_size, proj_dim, pool_out_len)
        pooled = self.pool(x)

        # 5) Flatten and final projection
        out = pooled.flatten(start_dim=1)           # (batch_size, proj_dim * pool_out_len)
        out = self.fc(out)                          # (batch_size, final_dim)
        return out


def get_inputs():
    """
    Builds realistic inputs for EmbeddingBag: concatenated indices and offsets.

    Returns:
        [indices_tensor, offsets_tensor]
    - indices_tensor: 1D torch.LongTensor with concatenated token indices
    - offsets_tensor: 1D torch.LongTensor with starting offsets for each sample
    """
    # Randomly choose bag lengths for each sample (at least 1 token per bag)
    lengths = torch.randint(low=1, high=max_bag_len + 1, size=(batch_size,), dtype=torch.long)
    total_tokens = int(lengths.sum().item())

    # Create random indices in [0, num_embeddings)
    indices = torch.randint(low=0, high=num_embeddings, size=(total_tokens,), dtype=torch.long)

    # Compute offsets: starting positions for each bag in the concatenated indices array
    offsets = torch.zeros(batch_size, dtype=torch.long)
    offsets[1:] = torch.cumsum(lengths, dim=0)[:-1]

    return [indices, offsets]


def get_init_inputs():
    """
    Returns initialization arguments for the Model constructor in the same order.

    Returns:
        [num_embeddings, embedding_dim, proj_dim, seq_len, pool_out_len, final_dim]
    """
    return [num_embeddings, embedding_dim, proj_dim, seq_len, pool_out_len, final_dim]