import torch
import torch.nn as nn

# Configuration variables (module level)
batch_size = 8
num_embeddings = 10000
embedding_dim = 128
bag_size = 16  # number of indices per bag
in_channels = 64  # will be the input channels for LazyConvTranspose1d (determined by the reshaped proj)
seq_len = 16  # temporal length before transposed convolution
conv_out_channels = 32  # desired output channels after ConvTranspose1d
kernel_size = 5
stride = 2
padding = 1
mode = 'mean'  # EmbeddingBag mode: 'mean' or 'sum' or 'max'


class Model(nn.Module):
    """
    A composite model that:
    - Uses an EmbeddingBag to aggregate sparse token indices into dense bag embeddings.
    - Projects the bag embedding into a 3D tensor (batch, in_channels, seq_len).
    - Applies a Hardsigmoid non-linearity then upsamples using LazyConvTranspose1d.
    - Uses a learned per-bag gate (also via Hardsigmoid) and a residual projected from the bag embedding,
      to produce the final output.

    The design demonstrates interaction between sparse embedding operations, dense linear projections,
    non-linearity (Hardsigmoid), and a lazily-initialized transposed convolution.
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        in_channels: int,
        seq_len: int,
        conv_out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        mode: str = 'mean'
    ):
        """
        Initializes submodules and precomputes the output temporal length of the transposed convolution.

        Args:
            num_embeddings: Vocabulary size for embeddings.
            embedding_dim: Dimension of each embedding vector.
            in_channels: Number of channels into the ConvTranspose1d (will be determined by the projected tensor).
            seq_len: Input temporal length to ConvTranspose1d.
            conv_out_channels: Number of output channels from ConvTranspose1d.
            kernel_size: Kernel size for ConvTranspose1d.
            stride: Stride for ConvTranspose1d.
            padding: Padding for ConvTranspose1d.
            mode: Mode for EmbeddingBag aggregation ('mean', 'sum', 'max').
        """
        super(Model, self).__init__()

        # Sparse aggregation
        self.embedding_bag = nn.EmbeddingBag(num_embeddings, embedding_dim, mode=mode)

        # Project embedding into a dense tensor for conv transpose input:
        # This maps (batch, embedding_dim) -> (batch, in_channels * seq_len)
        self.proj_to_seq = nn.Linear(embedding_dim, in_channels * seq_len)

        # Gate that modulates the conv output per-channel
        self.gate_proj = nn.Linear(embedding_dim, conv_out_channels)

        # Residual projection directly from embedding to match conv output shape
        # We'll compute seq_len_out and size the projection accordingly.
        self.seq_len = seq_len
        self.conv_out_channels = conv_out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Compute output length for ConvTranspose1d: L_out = (L_in - 1) * stride - 2*padding + kernel_size
        self.seq_len_out = (self.seq_len - 1) * self.stride - 2 * self.padding + self.kernel_size

        # Residual projection: (batch, embedding_dim) -> (batch, conv_out_channels * seq_len_out)
        self.res_proj = nn.Linear(embedding_dim, conv_out_channels * self.seq_len_out)

        # Non-linearity
        self.hsig = nn.Hardsigmoid()

        # Lazy conv transpose: out_channels is known, in_channels will be inferred on first forward
        self.convT = nn.LazyConvTranspose1d(
            out_channels=conv_out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

    def forward(self, indices: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that composes sparse aggregation, dense projections, non-linearity,
        transposed convolution, gating, and residual addition.

        Args:
            indices: 1D tensor of token indices concatenated for all bags.
            offsets: 1D tensor of offsets indicating bag starts (length == batch_size).

        Returns:
            Tensor of shape (batch_size, conv_out_channels, seq_len_out).
        """
        # 1) Aggregate sparse indices into bag embeddings: (batch_size, embedding_dim)
        bag_emb = self.embedding_bag(indices, offsets)  # (B, embedding_dim)

        # 2) Project to (B, in_channels * seq_len) and reshape to (B, in_channels, seq_len)
        seq_tensor = self.proj_to_seq(bag_emb)  # (B, in_channels * seq_len)
        seq_tensor = seq_tensor.view(-1, int(seq_tensor.shape[-1] // self.seq_len), self.seq_len)  # (B, in_channels, seq_len)

        # 3) Apply a bounded non-linearity (Hardsigmoid)
        seq_activated = self.hsig(seq_tensor)  # (B, in_channels, seq_len)

        # 4) Upsample / decode with transposed convolution -> (B, conv_out_channels, seq_len_out)
        conv_out = self.convT(seq_activated)

        # 5) Per-bag gating: project bag embedding to per-channel gates and apply Hardsigmoid
        gate = self.hsig(self.gate_proj(bag_emb))  # (B, conv_out_channels)
        gate = gate.unsqueeze(-1)  # (B, conv_out_channels, 1)
        gate = gate.expand(-1, -1, conv_out.shape[-1])  # broadcast across temporal dim

        gated = conv_out * gate  # (B, conv_out_channels, seq_len_out)

        # 6) Residual path from embedding, shaped to conv output and added
        residual = self.res_proj(bag_emb)  # (B, conv_out_channels * seq_len_out)
        residual = residual.view(-1, self.conv_out_channels, self.seq_len_out)  # (B, conv_out_channels, seq_len_out)

        out = torch.tanh(gated + residual)  # final bounded output

        return out


def get_inputs():
    """
    Returns the inputs expected by Model.forward:
    - indices: 1D tensor of length batch_size * bag_size
    - offsets: 1D tensor of bag start offsets of length batch_size
    """
    # Random indices for each bag
    indices = torch.randint(0, num_embeddings, (batch_size * bag_size,), dtype=torch.long)
    offsets = torch.arange(0, batch_size * bag_size, bag_size, dtype=torch.long)
    return [indices, offsets]


def get_init_inputs():
    """
    Returns initialization parameters to instantiate Model in the same order as Model.__init__.
    """
    return [
        num_embeddings,
        embedding_dim,
        in_channels,
        seq_len,
        conv_out_channels,
        kernel_size,
        stride,
        padding,
        mode
    ]