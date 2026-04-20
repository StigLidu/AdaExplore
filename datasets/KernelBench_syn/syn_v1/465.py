import torch
import torch.nn as nn
import torch.nn.functional as F

# Configuration / hyperparameters
NUM_BAGS = 8           # number of "bags" (acts like a batch size for EmbeddingBag)
AVG_BAG_LEN = 12       # fixed length for each bag (for simplicity in get_inputs)
NUM_EMBEDDINGS = 10000 # vocabulary size for embedding bag
EMBED_DIM = 64         # dimensionality of each embedding vector
HIDDEN_CHANNELS = 32   # number of channels produced by the dense projection (pre-deconv)
SEQ_LEN = 8            # initial sequence length after projection (will be upsampled)
OUT_CHANNELS = 16      # number of channels after ConvTranspose1d (deconv)
OUTPUT_DIM = 10        # final per-position output dimensionality (e.g., logits)

class Model(nn.Module):
    """
    Composite model that:
      - Aggregates token indices per-bag using EmbeddingBag (mean pooling).
      - Projects each bag embedding into a hidden sequence via a Linear layer.
      - Reshapes to (N, channels, seq_len) and applies a LazyConvTranspose1d to
        upsample the temporal dimension.
      - Normalizes and applies a pointwise Linear to get per-position outputs.

    Forward signature:
        forward(indices: Tensor, offsets: Tensor) -> Tensor
    where:
        indices: 1D LongTensor containing concatenated token indices for all bags.
        offsets: 1D LongTensor containing start indices for each bag (length = num_bags).
    """
    def __init__(
        self,
        num_embeddings: int,
        embed_dim: int,
        hidden_channels: int,
        out_channels: int,
        seq_len: int,
        output_dim: int
    ):
        super(Model, self).__init__()

        # EmbeddingBag aggregates variable-length sets of token indices into fixed-size vectors
        self.embedding_bag = nn.EmbeddingBag(num_embeddings=num_embeddings, embedding_dim=embed_dim, mode='mean')

        # Project each bag embedding into a flattened hidden sequence of shape (hidden_channels * seq_len)
        self.linear_proj = nn.Linear(embed_dim, hidden_channels * seq_len)

        # Store config for reshapes
        self.hidden_channels = hidden_channels
        self.seq_len = seq_len

        # LazyConvTranspose1d: in_channels will be inferred on first forward pass.
        # We set out_channels explicitly to control downstream channel dimensionality.
        # Kernel and stride chosen to double the sequence length: kernel_size=4, stride=2, padding=1
        self.deconv = nn.LazyConvTranspose1d(out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=True)

        # Normalize across channels for stability after deconvolution
        self.bn = nn.BatchNorm1d(out_channels)

        # Final pointwise projection mapping channels -> output_dim for each sequence position
        self.final_linear = nn.Linear(out_channels, output_dim)

        # Activation
        self.activation = nn.ReLU(inplace=True)

        # Initialize linear layers with a sensible default
        nn.init.xavier_uniform_(self.linear_proj.weight)
        if self.linear_proj.bias is not None:
            nn.init.zeros_(self.linear_proj.bias)
        nn.init.xavier_uniform_(self.final_linear.weight)
        if self.final_linear.bias is not None:
            nn.init.zeros_(self.final_linear.bias)

    def forward(self, indices: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the composite pipeline.

        Args:
            indices (torch.Tensor): 1D LongTensor of token indices (concatenated bags), shape (total_indices,).
            offsets (torch.Tensor): 1D LongTensor of bag offsets, shape (num_bags,).

        Returns:
            torch.Tensor: Output tensor of shape (num_bags, upsampled_seq_len, output_dim).
        """
        # 1) Aggregate tokens per bag -> (N, embed_dim)
        bag_emb = self.embedding_bag(indices, offsets)  # (num_bags, embed_dim)

        # 2) Project to a flattened hidden sequence -> (N, hidden_channels * seq_len)
        proj = self.linear_proj(bag_emb)
        proj = self.activation(proj)

        # 3) Reshape to sequence tensor -> (N, hidden_channels, seq_len)
        N = proj.size(0)
        hidden = proj.view(N, self.hidden_channels, self.seq_len)

        # 4) Upsample temporal dimension with ConvTranspose1d -> (N, out_channels, seq_len * 2)
        x = self.deconv(hidden)
        x = self.bn(x)
        x = self.activation(x)

        # 5) Permute to (N, seq_len_out, out_channels) to apply pointwise linear per position
        x = x.permute(0, 2, 1).contiguous()

        # 6) Final linear projection to per-position outputs -> (N, seq_len_out, output_dim)
        out = self.final_linear(x)

        return out

# Public API functions for the testing harness

def get_inputs():
    """
    Returns a list of input tensors to be passed to the Model.forward.
    We create NUM_BAGS bags, each with AVG_BAG_LEN indices, concatenated into a single 1D indices tensor,
    and offsets that mark the start of each bag.
    """
    total_indices = NUM_BAGS * AVG_BAG_LEN
    # Random token indices in range [0, NUM_EMBEDDINGS)
    indices = torch.randint(low=0, high=NUM_EMBEDDINGS, size=(total_indices,), dtype=torch.long)
    # Offsets: start positions for each bag (assume equal-length bags here for simplicity)
    offsets = torch.arange(0, total_indices, step=AVG_BAG_LEN, dtype=torch.long)
    return [indices, offsets]

def get_init_inputs():
    """
    Returns the initialization parameters for the Model constructor in the same order.
    """
    return [NUM_EMBEDDINGS, EMBED_DIM, HIDDEN_CHANNELS, OUT_CHANNELS, SEQ_LEN, OUTPUT_DIM]