import math
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Attention-like module that projects an input sequence into query/key/value spaces,
    computes scaled attention scores, applies a CELU nonlinearity to the scores,
    normalizes them with LogSoftmax, and finally gates the aggregated values with a Sigmoid.
    
    This combines elementwise non-linearities (CELU, Sigmoid) with a LogSoftmax normalization
    to form a compact, but non-trivial computation graph.
    """
    def __init__(self, d_model: int, d_hidden: int):
        super(Model, self).__init__()
        # Non-linearities used in the forward pass
        self.celu = nn.CELU(alpha=1.0)
        self.sigmoid = nn.Sigmoid()
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        
        # Internal scaling for attention scores
        self.scale = 1.0 / math.sqrt(d_hidden)
        
        # Note: weight matrices are provided as inputs in get_inputs() to mimic explicit parameters,
        # but the module still owns the nonlinear layers and scaling logic.
        self.d_model = d_model
        self.d_hidden = d_hidden

    def forward(
        self,
        x: torch.Tensor,
        W_q: torch.Tensor,
        W_k: torch.Tensor,
        W_v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            W_q: Query projection matrix of shape (d_model, d_hidden)
            W_k: Key projection matrix of shape (d_model, d_hidden)
            W_v: Value projection matrix of shape (d_model, d_hidden)

        Returns:
            out: Output tensor of shape (batch_size, seq_len, d_hidden)
        """
        # Project inputs to Q, K, V
        # Shapes: (B, S, H)
        Q = torch.matmul(x, W_q)
        K = torch.matmul(x, W_k)
        V = torch.matmul(x, W_v)

        # Compute scaled attention scores: (B, S, S)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Introduce a smooth nonlinearity to the scores before normalization
        scores = self.celu(scores)

        # Normalize with LogSoftmax for numerical stability, then exponentiate to recover probabilities
        attn_log = self.logsoftmax(scores)
        attn = torch.exp(attn_log)  # (B, S, S), same as softmax(scores) but via log route

        # Aggregate values
        out = torch.matmul(attn, V)  # (B, S, H)

        # Compute a gating vector from Q (global per-sequence statistics)
        gate = self.sigmoid(Q.mean(dim=1, keepdim=True))  # (B, 1, H)

        # Apply gating to the aggregated output (broadcast over sequence dimension)
        out = out * gate  # (B, S, H)

        return out

# Configuration / shapes
batch_size = 32
seq_len = 128
d_model = 256
d_hidden = 64

def get_inputs():
    """
    Returns inputs for the Model.forward:
    - x: (batch_size, seq_len, d_model)
    - W_q, W_k, W_v: (d_model, d_hidden)
    """
    # Use float32 for a mix of stability and performance on CPU/GPU
    x = torch.randn(batch_size, seq_len, d_model, dtype=torch.float32)
    W_q = torch.randn(d_model, d_hidden, dtype=torch.float32)
    W_k = torch.randn(d_model, d_hidden, dtype=torch.float32)
    W_v = torch.randn(d_model, d_hidden, dtype=torch.float32)
    return [x, W_q, W_k, W_v]

def get_init_inputs():
    """
    Returns initialization parameters for the Model constructor.
    """
    return [d_model, d_hidden]