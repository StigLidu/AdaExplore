import torch
import torch.nn as nn
from typing import Optional, List

class Model(nn.Module):
    """
    Multi-head attention inspired block that demonstrates a combination of:
    - Linear projections to Q/K/V
    - Scaled dot-product attention with Softmax
    - Non-linearity applied with LeakyReLU
    - Auxiliary LogSoftmax-based token weighting and residual fusion

    The module accepts an input tensor of shape (batch, seq_len, embed_dim) and an
    optional additive attention mask (shape broadcastable to (batch, num_heads, seq_len, seq_len))
    where masked positions should contain large negative values (e.g., -1e9).
    """
    def __init__(self, embed_dim: int, num_heads: int, negative_slope: float = 0.1):
        super(Model, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Projections for queries, keys and values
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Output projection and non-linearity
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.leaky = nn.LeakyReLU(negative_slope=negative_slope)

        # Softmax for attention weights and LogSoftmax for auxiliary token weighting
        self.softmax = nn.Softmax(dim=-1)
        self.logsoftmax = nn.LogSoftmax(dim=1)  # log-probs across sequence dimension

        # scaling factor for dot-product attention
        self.scale = float(self.head_dim) ** -0.5

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass:
        1. Project input to Q, K, V
        2. Compute scaled dot-product attention scores: scores = (Q K^T) * scale
        3. Optionally add an additive attention mask to scores
        4. Softmax over last dim to produce attention weights
        5. Compute context = attn_weights @ V
        6. Project context back and apply LeakyReLU
        7. Compute log-probabilities across tokens from a simple token scoring branch (auxiliary)
           and use them to produce a pooled vector
        8. Fuse pooled vector back into token-level context (residual style) and return

        Args:
            x: Tensor of shape (B, T, C)
            attn_mask: Optional additive mask (broadcastable to (B, num_heads, T, T)
                       or (T, T), containing large negative values where masking is needed)

        Returns:
            Tensor of shape (B, T, C)
        """
        B, T, C = x.shape
        # Project to Q, K, V and reshape for multi-head: (B, num_heads, T, head_dim)
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

        # Scaled dot-product attention scores: (B, num_heads, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply additive mask if provided (e.g., causal mask or padding mask with -1e9)
        if attn_mask is not None:
            # Allow attn_mask shape to be (T, T) or (B, T, T) or (B, 1, T, T)
            # Expand/broadcast to (B, num_heads, T, T) via unsqueeze where necessary
            mask = attn_mask
            if mask.dim() == 2:  # (T, T)
                mask = mask.unsqueeze(0).unsqueeze(0)  # (1,1,T,T)
            elif mask.dim() == 3:  # (B, T, T)
                mask = mask.unsqueeze(1)  # (B,1,T,T)
            # mask now broadcastable to scores
            scores = scores + mask

        # Attention weights and context computation
        attn_weights = self.softmax(scores)  # (B, num_heads, T, T)
        context = torch.matmul(attn_weights, v)  # (B, num_heads, T, head_dim)

        # Merge heads and project out
        context = context.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        context = self.out_proj(context)  # (B, T, C)

        # Non-linearity applied to projected context
        context = self.leaky(context)

        # Auxiliary branch: compute simple token scores and obtain a pooled vector
        # Use LogSoftmax across the sequence dimension to produce stable token weights
        token_scores = x.mean(dim=-1)  # (B, T) simple per-token scalar score
        log_probs = self.logsoftmax(token_scores)  # (B, T)
        weights = torch.exp(log_probs).unsqueeze(-1)  # (B, T, 1) normalized over T

        pooled = (context * weights).sum(dim=1)  # (B, C) weighted pooling

        # Fuse pooled vector back into token-level representations (residual-style)
        out = context + pooled.unsqueeze(1)  # (B, T, C)

        return out

# Top-level configuration for test inputs
batch_size = 8
seq_len = 128
embed_dim = 512
num_heads = 8

def get_inputs() -> List[torch.Tensor]:
    """
    Returns:
        [x, attn_mask]
        - x: random input tensor of shape (batch_size, seq_len, embed_dim)
        - attn_mask: causal additive mask of shape (seq_len, seq_len) with -1e9 above diagonal
                     (so it can be broadcast to (1,1,seq_len,seq_len) and used for causal attention)
    """
    x = torch.randn(batch_size, seq_len, embed_dim)
    # causal mask: prevent attending to future positions (upper triangular) by adding large negative values
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.float32), diagonal=1) * -1e9
    return [x, causal_mask]

def get_init_inputs() -> List[int]:
    """
    Returns initialization parameters required for Model construction:
        [embed_dim, num_heads]
    """
    return [embed_dim, num_heads]