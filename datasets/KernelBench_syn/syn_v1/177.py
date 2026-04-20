import torch
import torch.nn as nn

"""
Complex sequence processing module that combines ZeroPad1d, GRU, and TransformerDecoderLayer
to create a hybrid encoder-decoder style computation producing a per-batch output vector.

Structure:
- Model(nn.Module) with __init__ and forward
- get_inputs() producing test tensors
- get_init_inputs() returning initialization parameters (empty here)

Processing pattern:
1. Project input features into an embedding space.
2. Use ZeroPad1d to add left/right temporal padding (done by permuting to (batch, channels, seq_len)).
3. Run a multi-layer GRU over the padded sequence.
4. Project external memory/context into the same embedding dimension.
5. Run a TransformerDecoderLayer with the GRU outputs as the target and the projected memory as "memory".
6. Residual combine decoder output and GRU output, apply activation.
7. Pool across the (padded) sequence and project to final logits.
"""

# Configuration / shape constants
SEQ_LEN = 128        # original source sequence length (time steps)
BATCH = 16           # batch size
INPUT_DIM = 64       # input feature dimension per time step
MEM_LEN = 32         # memory/context sequence length
MEM_DIM = 96         # memory feature dimension per memory time step
EMBED_DIM = 128      # embedding dimension used throughout (d_model)
GRU_LAYERS = 2       # number of GRU layers
PAD_LEFT = 2         # left padding (time steps)
PAD_RIGHT = 3        # right padding (time steps)
NHEAD = 8            # number of attention heads for TransformerDecoderLayer
FFN_DIM = 512        # feedforward inner dimension in TransformerDecoderLayer
OUTPUT_DIM = 10      # final output dimension per batch element (e.g., classes)

class Model(nn.Module):
    """
    Hybrid sequence model combining ZeroPad1d, GRU and TransformerDecoderLayer.

    Inputs:
        src (torch.Tensor): Source sequence of shape (seq_len, batch, INPUT_DIM)
        memory (torch.Tensor): Memory/context sequence of shape (mem_len, batch, MEM_DIM)

    Returns:
        torch.Tensor: Output logits of shape (batch, OUTPUT_DIM)
    """
    def __init__(self):
        super(Model, self).__init__()
        # Project input features into embedding space expected by GRU / Transformer
        self.input_proj = nn.Linear(INPUT_DIM, EMBED_DIM)
        # Project memory/context into same embedding dimension
        self.mem_proj = nn.Linear(MEM_DIM, EMBED_DIM)
        # ZeroPad1d pads (left, right) on last dimension of (N, C, L) inputs.
        # We'll permute sequence tensors to (batch, embed_dim, seq_len) before applying.
        self.pad = nn.ZeroPad1d((PAD_LEFT, PAD_RIGHT))
        # GRU processes the (padded) embedded sequence. Set hidden size equal to EMBED_DIM
        # so outputs can be combined with Transformer outputs without extra projection.
        self.gru = nn.GRU(input_size=EMBED_DIM, hidden_size=EMBED_DIM, num_layers=GRU_LAYERS)
        # Single TransformerDecoderLayer: expects d_model == EMBED_DIM
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=EMBED_DIM, nhead=NHEAD, dim_feedforward=FFN_DIM)
        # Final projection from pooled embedding to desired output dimension
        self.fc_out = nn.Linear(EMBED_DIM, OUTPUT_DIM)
        # Small layer norm for stability after combining GRU + Decoder outputs
        self.layer_norm = nn.LayerNorm(EMBED_DIM)
        # Activation
        self.act = nn.ReLU()

    def forward(self, src: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining padding, GRU, and TransformerDecoderLayer.

        Args:
            src: Tensor of shape (seq_len, batch, INPUT_DIM)
            memory: Tensor of shape (mem_len, batch, MEM_DIM)

        Returns:
            logits: Tensor of shape (batch, OUTPUT_DIM)
        """
        # Step 1: Input projection to embedding space
        # -> (seq_len, batch, EMBED_DIM)
        embed_src = self.input_proj(src)

        # Step 2: ZeroPad1d expects (N, C, L). Convert (seq_len, batch, EMBED_DIM) ->
        # (batch, EMBED_DIM, seq_len)
        embed_src_bcL = embed_src.permute(1, 2, 0)  # (batch, embed_dim, seq_len)
        padded_bcL = self.pad(embed_src_bcL)       # (batch, embed_dim, seq_len + pad_left + pad_right)

        # Convert back to (padded_seq_len, batch, embed_dim) for GRU / Transformer
        padded_seq = padded_bcL.permute(2, 0, 1)   # (padded_seq_len, batch, embed_dim)

        # Step 3: GRU over the padded sequence
        # gru_out: (padded_seq_len, batch, EMBED_DIM)
        gru_out, _ = self.gru(padded_seq)

        # Step 4: Project memory/context into embedding space
        # memory: (mem_len, batch, MEM_DIM) -> mem_embed: (mem_len, batch, EMBED_DIM)
        mem_embed = self.mem_proj(memory)

        # Step 5: TransformerDecoderLayer: treat GRU outputs as 'tgt' and mem_embed as 'memory'
        # Both tensors have shapes (seq, batch, embed_dim) suitable for the layer.
        dec_out = self.decoder_layer(tgt=gru_out, memory=mem_embed)

        # Step 6: Residual connection + normalization + activation
        combined = self.layer_norm(dec_out + gru_out)
        combined = self.act(combined)

        # Step 7: Pool across the (padded) time dimension and project to final logits
        # combined: (padded_seq_len, batch, embed_dim) -> mean over time -> (batch, embed_dim)
        pooled = combined.mean(dim=0)
        logits = self.fc_out(pooled)  # (batch, OUTPUT_DIM)

        return logits

def get_inputs():
    """
    Generate random input tensors matching the configured shapes.

    Returns:
        list: [src, memory] where
            src: Tensor of shape (SEQ_LEN, BATCH, INPUT_DIM)
            memory: Tensor of shape (MEM_LEN, BATCH, MEM_DIM)
    """
    src = torch.randn(SEQ_LEN, BATCH, INPUT_DIM)
    memory = torch.randn(MEM_LEN, BATCH, MEM_DIM)
    return [src, memory]

def get_init_inputs():
    """
    No special initialization parameters needed for this model.

    Returns:
        list: Empty list.
    """
    return []