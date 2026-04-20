import math
import torch
import torch.nn as nn

# Configuration
batch_size = 8
seq_len = 128
emb_dim = 512
proj_dim = 256  # hidden / projection dimension for Q/K/V

class Model(nn.Module):
    """
    Attention-style module that:
    - Projects inputs to Q, K, V via provided weight matrices
    - Applies a CELU non-linearity to Q
    - Computes scaled dot-product attention with an optional mask
    - Pools the attended representations (mean and max), concatenates them
    - Produces a final distribution with LogSoftmax

    Initialization expects explicit weight tensors for the three projections:
        W_q: (emb_dim, proj_dim)
        W_k: (emb_dim, proj_dim)
        W_v: (emb_dim, proj_dim)
    """
    def __init__(self, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor):
        super(Model, self).__init__()
        # Register projection weights as parameters so the module is self-contained
        self.W_q = nn.Parameter(W_q)
        self.W_k = nn.Parameter(W_k)
        self.W_v = nn.Parameter(W_v)

        # Non-linearities / normalization layers used in the forward pass
        self.celu = nn.CELU()  # elementwise activation on queries
        self.softmax = nn.Softmax(dim=-1)  # attention distribution over sequence dimension
        self.logsoftmax = nn.LogSoftmax(dim=-1)  # final normalized log-probabilities

        # Precompute scale for dot-product attention
        # Use float to avoid integer sqrt issues
        self.scale = math.sqrt(float(self.W_q.size(1)))

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, S, D) where
                              B=batch_size, S=seq_len, D=emb_dim.
            src_mask (torch.Tensor or None): Optional boolean mask of shape (B, S).
                              True indicates valid tokens; False indicates masked tokens.

        Returns:
            torch.Tensor: Log-softmaxed vector of shape (B, 2 * proj_dim)
                          produced by pooling the attended representations.
        """
        # x: (B, S, D)
        # Project to Q, K, V using matmul with the provided weight matrices
        # Results are (B, S, H)
        Q = torch.matmul(x, self.W_q)  # (B, S, H)
        K = torch.matmul(x, self.W_k)  # (B, S, H)
        V = torch.matmul(x, self.W_v)  # (B, S, H)

        # Apply non-linearity to queries to introduce elementwise gating
        Q = self.celu(Q)  # (B, S, H)

        # Scaled dot-product attention scores: (B, S, S)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # If a source mask is provided (B, S), expand to (B, S, S) and mask out invalid positions
        if src_mask is not None:
            # Ensure boolean mask
            mask_bool = src_mask.bool()  # (B, S)
            # Expand to (B, 1, S) then broadcast to (B, S, S) where each query position sees the same key mask
            key_mask = mask_bool.unsqueeze(1).expand(-1, x.size(1), -1)  # (B, S, S)
            scores = scores.masked_fill(~key_mask, float('-1e9'))

        # Attention weights and context
        attn_weights = self.softmax(scores)  # (B, S, S)
        context = torch.matmul(attn_weights, V)  # (B, S, H)

        # Pooling: mean and max across sequence dimension
        pooled_mean = context.mean(dim=1)  # (B, H)
        pooled_max = context.max(dim=1).values  # (B, H)

        # Concatenate pooled summaries and produce a log-probability vector
        combined = torch.cat([pooled_mean, pooled_max], dim=-1)  # (B, 2H)
        out = self.logsoftmax(combined)  # (B, 2H)

        return out

def get_inputs():
    """
    Generates a random input batch and a random boolean source mask.
    Mask True indicates a valid token; ~10% of positions are masked out.
    """
    x = torch.randn(batch_size, seq_len, emb_dim)
    # Create a mask with approximately 10% masked positions
    src_mask = (torch.rand(batch_size, seq_len) > 0.1)
    return [x, src_mask]

def get_init_inputs():
    """
    Returns initialization weight tensors for W_q, W_k, W_v.
    Shapes:
      W_*: (emb_dim, proj_dim)
    These are intended to be passed into Model(...).
    """
    W_q = torch.randn(emb_dim, proj_dim)
    W_k = torch.randn(emb_dim, proj_dim)
    W_v = torch.randn(emb_dim, proj_dim)
    return [W_q, W_k, W_v]