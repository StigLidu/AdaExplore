import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A composite module that:
    - Projects source and target embeddings into a higher-dimensional space,
    - Applies GLU gating to produce model-sized representations,
    - Uses Local Response Normalization across the channel dimension to enforce local competition,
    - Runs a stack of TransformerDecoder layers to attend target to source (memory),
    - Fuses decoder output with the normalized target via another GLU and produces a pooled output.

    Inputs:
        src: Tensor of shape (src_seq_len, batch_size, embed_dim)
        tgt: Tensor of shape (tgt_seq_len, batch_size, embed_dim)

    Output:
        Tensor of shape (batch_size, embed_dim) -- pooled and projected output per batch element.
    """
    def __init__(
        self,
        embed_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        lrn_size: int,
    ):
        super(Model, self).__init__()
        # Linear projections produce 2 * d_model so GLU can halve it to d_model
        self.src_proj = nn.Linear(embed_dim, 2 * d_model)
        self.tgt_proj = nn.Linear(embed_dim, 2 * d_model)

        # GLU layers to gate information after linear projection and for fusion
        self.glu_proj = nn.GLU(dim=-1)    # used for initial projections
        self.glu_fuse = nn.GLU(dim=-1)    # used for fusing decoder output with target residual

        # Local Response Normalization operates across the channel dimension.
        # We will permute (seq_len, batch, d_model) -> (batch, d_model, seq_len) for this.
        # alpha, beta, k set to reasonable defaults; lrn_size provided as parameter.
        self.lrn = nn.LocalResponseNorm(size=lrn_size, alpha=1e-3, beta=0.75, k=1.0)

        # Transformer decoder stack: decoder layers attend tgt -> memory (src)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation="relu",
            batch_first=False,  # we'll use (seq_len, batch, d_model)
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Final projection mapping pooled d_model back to embed_dim
        self.output_linear = nn.Linear(d_model, embed_dim)

        # Keep dims for potential sanity checks or downstream uses
        self.d_model = d_model

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining projections, GLU gating, LRN, Transformer decoding, fusion, and pooling.

        Args:
            src: Source tensor of shape (src_seq_len, batch_size, embed_dim)
            tgt: Target tensor of shape (tgt_seq_len, batch_size, embed_dim)

        Returns:
            out: Tensor of shape (batch_size, embed_dim)
        """
        # Project and apply GLU to obtain (seq_len, batch, d_model)
        src_proj = self.src_proj(src)         # (src_seq_len, batch, 2*d_model)
        src_glu = self.glu_proj(src_proj)     # (src_seq_len, batch, d_model)

        tgt_proj = self.tgt_proj(tgt)         # (tgt_seq_len, batch, 2*d_model)
        tgt_glu = self.glu_proj(tgt_proj)     # (tgt_seq_len, batch, d_model)

        # Apply Local Response Normalization across the channel dimension.
        # Permute to (batch, d_model, seq_len) -> apply LRN -> back to (seq_len, batch, d_model)
        src_norm = self.lrn(src_glu.permute(1, 2, 0)).permute(2, 0, 1)
        tgt_norm = self.lrn(tgt_glu.permute(1, 2, 0)).permute(2, 0, 1)

        # Use the TransformerDecoder to let target attend to source (as memory).
        # Both inputs must be (seq_len, batch, d_model)
        decoder_out = self.decoder(tgt_norm, src_norm)  # (tgt_seq_len, batch, d_model)

        # Fuse decoder output with the normalized target via concatenation and GLU gating.
        # Concatenate along the feature dimension -> (tgt_seq_len, batch, 2*d_model)
        fused = torch.cat([decoder_out, tgt_norm], dim=-1)
        fused_gated = self.glu_fuse(fused)  # (tgt_seq_len, batch, d_model)

        # Pool across the sequence dimension (mean pooling) to produce (batch, d_model)
        pooled = fused_gated.mean(dim=0)

        # Final projection back to embed_dim
        out = self.output_linear(pooled)  # (batch, embed_dim)
        return out

# Module-level configuration variables
batch_size = 12
src_seq_len = 50
tgt_seq_len = 30
embed_dim = 128   # input embedding size for src and tgt
d_model = 256     # internal model dimension for transformer
nhead = 8
num_layers = 3
dim_feedforward = 1024
lrn_size = 5      # number of neighboring channels for LocalResponseNorm

def get_inputs():
    """
    Returns:
        [src, tgt] where:
            src: Tensor of shape (src_seq_len, batch_size, embed_dim)
            tgt: Tensor of shape (tgt_seq_len, batch_size, embed_dim)
    """
    src = torch.randn(src_seq_len, batch_size, embed_dim)
    tgt = torch.randn(tgt_seq_len, batch_size, embed_dim)
    return [src, tgt]

def get_init_inputs():
    """
    Returns initialization arguments for Model(...) in the same order as its signature.
    """
    return [embed_dim, d_model, nhead, num_layers, dim_feedforward, lrn_size]