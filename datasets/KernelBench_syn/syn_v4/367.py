import torch
import torch.nn as nn
import math
from typing import List

# Configuration
BATCH_SIZE = 8
TGT_SEQ_LEN = 128
MEM_SEQ_LEN = 64
D_MODEL = 512
NHEAD = 8
NUM_LAYERS = 4
DIM_FEEDFORWARD = 2048
DROPOUT = 0.1
BATCH_FIRST = True  # Use (batch, seq, feature) layout

class Model(nn.Module):
    """
    A composite model that:
      - Adds sinusoidal positional encodings to target tokens
      - Runs a stack of TransformerDecoder layers over the target using a memory tensor
      - Applies LazyInstanceNorm1d over the feature channels per sample
      - Projects back with a linear layer and adds a residual connection

    The model demonstrates interplay between transformer decoding and 1D instance normalization.
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = DIM_FEEDFORWARD,
        dropout: float = DROPOUT,
        batch_first: bool = BATCH_FIRST
    ):
        super(Model, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.batch_first = batch_first

        # Single decoder layer prototype with batch_first option
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=batch_first
        )
        # Stack of decoder layers
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Lazy InstanceNorm1d will infer num_features on first forward when given (N, C, L)
        self.norm = nn.LazyInstanceNorm1d(affine=True)

        # Final projection (keeps dimensionality but can be used to mix features)
        self.out_proj = nn.Linear(d_model, d_model)

        # Small dropout between operations
        self.dropout = nn.Dropout(dropout)

    def _positional_encoding(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generate sinusoidal positional encodings (seq_len, d_model)
        Returns a tensor shaped (1, seq_len, d_model) so it can be added to (batch, seq_len, d_model).
        """
        position = torch.arange(0, seq_len, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32, device=device) *
            -(math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(seq_len, self.d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, seq_len, d_model)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tgt: target tensor of shape (batch, tgt_seq_len, d_model)
            memory: memory tensor from encoder of shape (batch, mem_seq_len, d_model)

        Returns:
            Tensor of shape (batch, tgt_seq_len, d_model)
        """
        # 1) Add positional encoding to target
        pe = self._positional_encoding(tgt.size(1), tgt.device)
        tgt_pe = tgt + pe  # (batch, tgt_seq, d_model)

        # 2) Run through TransformerDecoder stack
        # The decoder expects (batch, seq, feature) when batch_first=True
        decoded = self.decoder(tgt_pe, memory)  # (batch, tgt_seq, d_model)
        decoded = self.dropout(decoded)

        # 3) Apply InstanceNorm across the channel dimension (convert to N, C, L)
        # Transformer output is (batch, seq, d_model) -> permute to (batch, d_model, seq)
        decoded_perm = decoded.permute(0, 2, 1)  # (batch, d_model, seq)
        normalized = self.norm(decoded_perm)     # LazyInstanceNorm1d infers num_features here
        normalized = normalized.permute(0, 2, 1) # back to (batch, seq, d_model)

        # 4) Linear projection and residual connection with original target (pre-positional)
        projected = self.out_proj(normalized)  # (batch, seq, d_model)

        # Ensure shapes align for residual; if they do not, use a simple scaling residual
        if projected.shape == tgt.shape:
            return projected + tgt  # residual connection
        else:
            return projected + 0.1 * tgt

# Input generation
def get_inputs() -> List[torch.Tensor]:
    """
    Returns:
        [tgt, memory] tensors where:
          - tgt has shape (BATCH_SIZE, TGT_SEQ_LEN, D_MODEL)
          - memory has shape (BATCH_SIZE, MEM_SEQ_LEN, D_MODEL)
    """
    tgt = torch.randn(BATCH_SIZE, TGT_SEQ_LEN, D_MODEL)
    memory = torch.randn(BATCH_SIZE, MEM_SEQ_LEN, D_MODEL)
    return [tgt, memory]

def get_init_inputs() -> List[int]:
    """
    Returns the initialization arguments for Model:
        [d_model, nhead, num_layers]
    """
    return [D_MODEL, NHEAD, NUM_LAYERS]