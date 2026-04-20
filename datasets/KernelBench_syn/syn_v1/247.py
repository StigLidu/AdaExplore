import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex module that combines linear projections, GLU gating, Hardswish/Hardtanh activations,
    and two batched matrix multiplications to create an attention-like mixing pattern.

    Computation summary:
      1. Project input into a doubled-hidden space and apply GLU to gate activations.
      2. Apply Hardswish non-linearity.
      3. Compute a self-similarity matrix via batched matmul (y @ y^T) and non-linearly modulate it.
      4. Re-apply similarities to the representations (another batched matmul).
      5. Reduce dimensionality through a linear layer, apply Hardswish + Hardtanh clamps.
      6. Project back to input embedding dim and add residual connection.
    """
    def __init__(self):
        super(Model, self).__init__()
        # Linear that outputs 2 * hidden_dim for GLU gating
        self.proj = nn.Linear(EMBED_DIM, HIDDEN_DIM * 2, bias=True)
        # GLU will split the last dimension in half and apply element-wise gating
        self.glu = nn.GLU(dim=-1)
        # Element-wise activations
        self.hardswish = nn.Hardswish()
        self.hardtanh = nn.Hardtanh(min_val=-1.0, max_val=1.0)
        # Bottleneck and output projections
        self.fc = nn.Linear(HIDDEN_DIM, PROJ_OUT, bias=True)
        self.out_proj = nn.Linear(PROJ_OUT, EMBED_DIM, bias=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            Tensor of same shape as x (batch_size, seq_len, embed_dim)
        """
        # Initial projection and gating
        # x -> (B, S, 2*H) -> GLU -> (B, S, H)
        y = self.proj(x)
        y = self.glu(y)

        # Non-linear transform
        y = self.hardswish(y)

        # Compute similarity/attention-like matrix
        # y: (B, S, H), k = y.transpose(1, 2): (B, H, S)
        # attn_scores: (B, S, S)
        k = y.transpose(1, 2)
        # Scale by sqrt(HIDDEN_DIM) for stability
        scale = math.sqrt(HIDDEN_DIM)
        attn_scores = torch.matmul(y, k) / scale

        # Non-linearly modulate scores to introduce sparsity & bounds
        attn_scores = self.hardswish(attn_scores)

        # Re-apply the similarity to mix representations
        # mixed: (B, S, H) = attn_scores (B, S, S) @ y (B, S, H)
        mixed = torch.matmul(attn_scores, y)

        # Bottleneck projection
        proj = self.fc(mixed)
        proj = self.hardswish(proj)
        proj = self.hardtanh(proj)  # clamp values to [-1, 1] for numerical stability

        # Project back to embedding dimension and add residual connection
        out = self.out_proj(proj)
        out = out + x  # residual

        return out

# Configuration variables
BATCH_SIZE = 4
SEQ_LEN = 128
EMBED_DIM = 768
HIDDEN_DIM = 1536
PROJ_OUT = 512

def get_inputs():
    """
    Returns a list containing a single input tensor matching the model's expected input shape.
    """
    x = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM)
    return [x]

def get_init_inputs():
    """
    No special initialization inputs required for this model (weights are randomly initialized).
    """
    return []