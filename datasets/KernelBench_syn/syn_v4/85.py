import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex model that uses InstanceNorm1d, a TransformerDecoder stack, and Sigmoid gating.
    - Normalizes source and target sequences with InstanceNorm1d (per-sample, per-channel).
    - Projects inputs to a shared embedding space, adds learned positional encodings.
    - Runs a TransformerDecoder to attend target tokens over source memory.
    - Applies a sigmoid gate (learned projection + Sigmoid) to modulate decoder outputs.
    - Produces a pooled output through mean-pooling and a final projection.

    Inputs:
        src: Tensor of shape (B, S, in_dim)
        tgt: Tensor of shape (B, T, in_dim)

    Output:
        Tensor of shape (B, out_dim)
    """
    def __init__(self,
                 in_dim: int,
                 embed_dim: int,
                 nhead: int,
                 num_layers: int,
                 max_src_len: int,
                 max_tgt_len: int,
                 out_dim: int,
                 dropout: float = 0.1):
        super(Model, self).__init__()

        # Normalization modules operate on (N, C, L) where C is feature dim
        self.src_instnorm = nn.InstanceNorm1d(in_dim, affine=True)
        self.tgt_instnorm = nn.InstanceNorm1d(in_dim, affine=True)

        # Input projections from raw feature dim to transformer embedding dim
        self.src_proj = nn.Linear(in_dim, embed_dim)
        self.tgt_proj = nn.Linear(in_dim, embed_dim)

        # Positional encodings (learned)
        self.src_pos_emb = nn.Parameter(torch.randn(max_src_len, embed_dim))
        self.tgt_pos_emb = nn.Parameter(torch.randn(max_tgt_len, embed_dim))

        # Transformer decoder stack
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, dropout=dropout, dim_feedforward=embed_dim * 4)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Gate projection and activation (uses nn.Sigmoid as requested)
        self.gate_proj = nn.Linear(embed_dim, embed_dim)
        self.gate_act = nn.Sigmoid()

        # Final projection after temporal pooling
        self.output_proj = nn.Linear(embed_dim, out_dim)

        # Small residual projection to match dims if needed
        self.residual_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            src: (B, S, in_dim)
            tgt: (B, T, in_dim)

        Returns:
            out: (B, out_dim)
        """
        B, S, _ = src.shape
        _, T, _ = tgt.shape

        # 1) Instance normalization expects (N, C, L) -> permute accordingly
        # Normalize across temporal dimension independently per sample & channel
        src_norm = self.src_instnorm(src.permute(0, 2, 1)).permute(0, 2, 1)  # (B, S, in_dim)
        tgt_norm = self.tgt_instnorm(tgt.permute(0, 2, 1)).permute(0, 2, 1)  # (B, T, in_dim)

        # 2) Linear projections into embedding space
        src_emb = self.src_proj(src_norm)  # (B, S, embed_dim)
        tgt_emb = self.tgt_proj(tgt_norm)  # (B, T, embed_dim)

        # 3) Add learned positional encodings (slice to actual lengths)
        src_pos = self.src_pos_emb[:S, :].unsqueeze(0)  # (1, S, embed_dim)
        tgt_pos = self.tgt_pos_emb[:T, :].unsqueeze(0)  # (1, T, embed_dim)
        src_emb = src_emb + src_pos
        tgt_emb = tgt_emb + tgt_pos

        # 4) Prepare for Transformer: (seq_len, batch, embed_dim)
        memory = src_emb.transpose(0, 1)  # (S, B, embed_dim)
        tgt_seq = tgt_emb.transpose(0, 1)  # (T, B, embed_dim)

        # 5) Decoder attends tgt over memory
        # Use no masks for simplicity; the decoder layer will apply causal masks if provided externally.
        dec_out = self.transformer_decoder(tgt_seq, memory)  # (T, B, embed_dim)

        # 6) Back to (B, T, E)
        dec_out = dec_out.transpose(0, 1)  # (B, T, embed_dim)
        dec_out = self.dropout(dec_out)

        # 7) Compute gating values from decoder outputs (element-wise modulation)
        gate = self.gate_act(self.gate_proj(dec_out))  # (B, T, embed_dim)
        gated = dec_out * gate  # (B, T, embed_dim)

        # 8) Residual connection: add projected target embeddings (stabilize signal)
        residual = self.residual_proj(tgt_emb)  # (B, T, embed_dim)
        combined = F.relu(gated + residual)  # (B, T, embed_dim)

        # 9) Temporal pooling and final projection
        pooled = combined.mean(dim=1)  # mean over time -> (B, embed_dim)
        out = self.output_proj(pooled)  # (B, out_dim)

        return out

# Configuration variables
BATCH = 8
SRC_LEN = 128
TGT_LEN = 64
IN_DIM = 512
EMBED_DIM = 256
NHEAD = 8
NUM_LAYERS = 4
OUT_DIM = 1024

def get_inputs():
    """
    Generates random source and target tensors for the model.

    Returns:
        list: [src, tgt] where
            src has shape (BATCH, SRC_LEN, IN_DIM)
            tgt has shape (BATCH, TGT_LEN, IN_DIM)
    """
    src = torch.randn(BATCH, SRC_LEN, IN_DIM)
    tgt = torch.randn(BATCH, TGT_LEN, IN_DIM)
    return [src, tgt]

def get_init_inputs():
    """
    Returns initialization parameters used to construct the model instance.
    These are provided separately so external harnesses can instantiate the model consistently.

    Returns:
        list: Arguments for Model(...) constructor in order.
    """
    return [IN_DIM, EMBED_DIM, NHEAD, NUM_LAYERS, SRC_LEN, TGT_LEN, OUT_DIM]