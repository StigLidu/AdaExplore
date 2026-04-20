import torch
import torch.nn as nn

# Configuration variables
batch_size = 8
tgt_seq_len = 32
memory_seq_len = 48

d_model = 256
nhead = 8
num_layers = 4
dim_feedforward = 1024
dropout = 0.1

class Model(nn.Module):
    """
    Decoder-based module that combines a stacked TransformerDecoder with a GLU-based
    gating projection. The model accepts a target sequence and a memory (encoder output),
    processes them through the TransformerDecoder stack, then applies a learned GLU gating
    and a residual projection with layer normalization.

    Inputs:
        tgt: Tensor of shape (T, N, d_model) -- target sequence (sequence length first)
        memory: Tensor of shape (S, N, d_model) -- memory/encoder outputs

    Output:
        Tensor of shape (T, N, d_model) -- processed decoder outputs
    """
    def __init__(
        self,
        d_model: int = d_model,
        nhead: int = nhead,
        num_layers: int = num_layers,
        dim_feedforward: int = dim_feedforward,
        dropout: float = dropout,
    ):
        super(Model, self).__init__()
        # Base Transformer decoder stack
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu"
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Input projections (identity-size for simplicity, but kept as Linear for flexibility)
        self.tgt_proj = nn.Linear(d_model, d_model)
        self.memory_proj = nn.Linear(d_model, d_model)

        # GLU-based gating: project to 2*d_model then apply GLU to get d_model outputs
        self.glu_linear = nn.Linear(d_model, d_model * 2)
        self.glu = nn.GLU(dim=-1)

        # Output projection and normalization with a residual connection
        self.out_proj = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
          1. Project target and memory into model dimension.
          2. Feed through TransformerDecoder stack.
          3. Apply a GLU gating projection to modulate decoder outputs.
          4. Apply a final linear projection and residual add with LayerNorm.

        Args:
            tgt: (T, N, d_model)
            memory: (S, N, d_model)

        Returns:
            out: (T, N, d_model)
        """
        # Project inputs (keeps the interface flexible if input dims change)
        tgt_p = self.tgt_proj(tgt)          # (T, N, d_model)
        mem_p = self.memory_proj(memory)    # (S, N, d_model)

        # Decode: attention over memory and self-attention among tgt tokens
        dec_out = self.decoder(tgt_p, mem_p)  # (T, N, d_model)

        # GLU gating: produce gates from decoder outputs and apply
        gated = self.glu(self.glu_linear(dec_out))  # (T, N, d_model)

        # Final projection and residual + normalization
        projected = self.out_proj(gated)  # (T, N, d_model)
        out = self.layer_norm(projected + dec_out)

        return out

def get_inputs():
    """
    Returns a list with:
      - tgt: Tensor of shape (tgt_seq_len, batch_size, d_model)
      - memory: Tensor of shape (memory_seq_len, batch_size, d_model)
    """
    tgt = torch.randn(tgt_seq_len, batch_size, d_model)
    memory = torch.randn(memory_seq_len, batch_size, d_model)
    return [tgt, memory]

def get_init_inputs():
    """
    Returns initialization parameters in the same order as Model.__init__ signature.
    """
    return [d_model, nhead, num_layers, dim_feedforward, dropout]