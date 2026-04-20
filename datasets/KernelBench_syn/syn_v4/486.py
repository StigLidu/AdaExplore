import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Sequence-to-sequence style module that combines LazyInstanceNorm1d, MultiheadAttention and a TransformerDecoder.
    The model normalizes source and target sequences per-channel (feature), performs a pooled cross-attention
    from the target to a pooled source representation, applies a residual connection, and then refines the
    target via a TransformerDecoder stack. A final projection and normalization is applied before returning.
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        """
        Initializes submodules.

        Args:
            d_model (int): Embedding dimensionality / number of features.
            nhead (int): Number of attention heads.
            num_layers (int): Number of decoder layers in the TransformerDecoder.
            dim_feedforward (int): Inner dimension of the transformer's feedforward network.
            dropout (float): Dropout probability for attention and feedforward layers.
        """
        super(Model, self).__init__()

        # LazyInstanceNorm1d will infer num_features (d_model) at first forward pass.
        # It expects input shaped (batch, channels, length).
        self.inst_norm = nn.LazyInstanceNorm1d()

        # A small cross-attention module that attends from target tokens to a pooled source representation.
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)

        # Build a TransformerDecoder stack to refine the attended target sequence using full source memory.
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Final linear projection to mix channels and introduce a learned residual transform.
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            src (torch.Tensor): Source sequence tensor of shape (batch_size, src_len, d_model).
            tgt (torch.Tensor): Target sequence tensor of shape (batch_size, tgt_len, d_model).

        Returns:
            torch.Tensor: Refined target tensor of shape (batch_size, tgt_len, d_model).
        """
        # 1) Channel-wise instance normalization across sequence length for both src and tgt.
        # LazyInstanceNorm1d expects (batch, channels, length), so transpose accordingly.
        src_norm = self.inst_norm(src.transpose(1, 2)).transpose(1, 2)  # (B, S, D)
        tgt_norm = self.inst_norm(tgt.transpose(1, 2)).transpose(1, 2)  # (B, T, D)

        # 2) Prepare sequences for modules that expect (seq_len, batch, embed_dim)
        memory = src_norm.transpose(0, 1)  # (S, B, D)
        tgt_seq = tgt_norm.transpose(0, 1)  # (T, B, D)

        # 3) Compute a pooled source representation (single-position memory) by averaging over source length.
        #    This produces a compact key/value for a focused cross-attention.
        memory_pool = memory.mean(dim=0, keepdim=True)  # (1, B, D)

        # 4) Multihead cross-attention from target positions to the pooled source vector.
        #    Result has shape (T, B, D).
        attn_output, _ = self.cross_attn(query=tgt_seq, key=memory_pool, value=memory_pool)

        # 5) Residual connection between normalized target and attention output.
        tgt_residual = tgt_seq + attn_output  # (T, B, D)

        # 6) Transformer decoder refines the target using the full source memory.
        decoded = self.decoder(tgt_residual, memory)  # (T, B, D)

        # 7) Back to (B, T, D), apply final projection and a channel-wise normalization.
        decoded = decoded.transpose(0, 1)  # (B, T, D)
        decoded = self.out_proj(decoded)  # (B, T, D)
        output = self.inst_norm(decoded.transpose(1, 2)).transpose(1, 2)  # (B, T, D)

        return output

# Configuration variables
batch_size = 8
src_len = 32
tgt_len = 16
d_model = 128
nhead = 8
num_layers = 3
dim_feedforward = 512
dropout = 0.1

def get_inputs():
    """
    Generates random source and target sequences for testing.

    Returns:
        list: [src, tgt] where
              src is (batch_size, src_len, d_model)
              tgt is (batch_size, tgt_len, d_model)
    """
    src = torch.randn(batch_size, src_len, d_model)
    tgt = torch.randn(batch_size, tgt_len, d_model)
    return [src, tgt]

def get_init_inputs():
    """
    Initialization parameters for the Model constructor.

    Returns:
        list: [d_model, nhead, num_layers, dim_feedforward, dropout]
    """
    return [d_model, nhead, num_layers, dim_feedforward, dropout]