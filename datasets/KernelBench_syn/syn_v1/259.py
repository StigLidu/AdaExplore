import torch
import torch.nn as nn
import math

# Configuration
BATCH_SIZE = 16
SEQ_LEN = 128
DIM = 512
HEADS = 8
HEAD_DIM = DIM // HEADS  # must divide evenly


class Model(nn.Module):
    """
    Attention-like block that combines linear projections, learned multi-head
    scaled dot-product attention, and a sequence of elementwise shrink/sigmoid/sign
    activations to produce a transformed output. This module intentionally uses
    non-standard attention weighting by gating softmax weights with a Hardsigmoid
    and then applies Softshrink as a sparse residual pathway. Softsign is used
    as the final nonlinearity before a LayerNorm.
    """
    def __init__(self):
        super(Model, self).__init__()
        # Linear projections for q,k,v and output projection
        self.q_proj = nn.Linear(DIM, DIM, bias=False)
        self.k_proj = nn.Linear(DIM, DIM, bias=False)
        self.v_proj = nn.Linear(DIM, DIM, bias=False)
        self.out_proj = nn.Linear(DIM, DIM, bias=False)

        # Activation modules from the provided list
        self.softshrink = nn.Softshrink(lambd=0.5)   # promotes sparsity in residual
        self.hardsigmoid = nn.Hardsigmoid()          # gates attention scores
        self.softsign = nn.Softsign()                # smooth final nonlinearity

        # Normalization for stability
        self.norm = nn.LayerNorm(DIM)

        # small epsilon to avoid division by zero
        self._eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, S, D)

        Returns:
            Tensor of shape (B, S, D) after attention-like mixing, nonlinearities,
            and a sparse residual pathway.
        """
        B, S, D = x.shape
        # Project inputs to q,k,v
        q = self.q_proj(x)  # (B, S, D)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head: (B, H, S, head_dim)
        q = q.view(B, S, HEADS, HEAD_DIM).transpose(1, 2)  # (B, H, S, head_dim)
        k = k.view(B, S, HEADS, HEAD_DIM).transpose(1, 2)
        v = v.view(B, S, HEADS, HEAD_DIM).transpose(1, 2)

        # Scaled dot-product attention scores: (B, H, S, S)
        scale = 1.0 / math.sqrt(HEAD_DIM)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Standard softmax attention weights
        attn_soft = torch.softmax(scores, dim=-1)

        # Gate the softmax weights with a hardsigmoid of the raw scores (values in [0,1])
        gate = self.hardsigmoid(scores)
        gated_attn = attn_soft * gate

        # Re-normalize gated attention weights to sum to 1 along last dim
        denom = gated_attn.sum(dim=-1, keepdim=True) + self._eps
        attn_weights = gated_attn / denom

        # Weighted sum to produce attention output: (B, H, S, head_dim)
        attn_out = torch.matmul(attn_weights, v)

        # Merge heads back: (B, S, D)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)

        # Linear projection out
        out = self.out_proj(attn_out)

        # Apply a smooth nonlinearity
        out = self.softsign(out)

        # Sparse residual pathway: apply softshrink to input (promotes sparsity)
        residual = self.softshrink(x)

        # Combine and normalize
        out = out + residual
        out = self.norm(out)

        return out


def get_inputs():
    """
    Returns:
        list: Single-element list with input tensor of shape (BATCH_SIZE, SEQ_LEN, DIM)
    """
    x = torch.randn(BATCH_SIZE, SEQ_LEN, DIM)
    return [x]


def get_init_inputs():
    """
    No special initialization parameters are required for this module.
    """
    return []