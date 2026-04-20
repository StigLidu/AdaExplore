import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex model combining linear projections, SELU activation,
    SyncBatchNorm over the hidden channels, and a gating mechanism
    using LeakyReLU. The model operates on sequence data of shape
    (batch, seq_len, in_dim) and returns an output of shape
    (batch, seq_len, out_dim).
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, negative_slope: float = 0.1):
        """
        Initializes layers:
          - fc1: project input features to hidden_dim
          - selu: non-linearity applied elementwise
          - bn: SyncBatchNorm applied over hidden channels (expects input permuted to (N, C, L))
          - fc2: project back to out_dim per position
          - gate_proj: projects pooled features to out_dim for gating
          - leaky: LeakyReLU used in gating pathway

        Args:
            in_dim (int): Input feature dimensionality.
            hidden_dim (int): Hidden channel dimensionality used for normalization.
            out_dim (int): Output feature dimensionality per sequence position.
            negative_slope (float): Negative slope for LeakyReLU.
        """
        super(Model, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=True)
        self.selu = nn.SELU()
        # SyncBatchNorm expects the channel dimension as C; we will permute (N, L, C) -> (N, C, L)
        self.bn = nn.SyncBatchNorm(num_features=hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=True)
        self.gate_proj = nn.Linear(out_dim, out_dim, bias=True)
        self.leaky = nn.LeakyReLU(negative_slope=negative_slope)

        # Initialize linear layers with a stable initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
          1. Linear projection to hidden_dim per position
          2. SELU activation
          3. Permute and apply SyncBatchNorm across channels
          4. Permute back and project to out_dim per position
          5. Global (temporal) mean pooling over sequence dimension
          6. Apply LeakyReLU to pooled vector and project to gating logits
          7. Sigmoid gating applied per-channel and broadcast across sequence positions

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, in_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, out_dim)
        """
        # 1 -> 2: position-wise projection and non-linearity
        y = self.fc1(x)                      # (B, L, hidden_dim)
        y = self.selu(y)                     # (B, L, hidden_dim)

        # 3: Batch-Normalize over channels: permute to (B, C, L)
        y_perm = y.permute(0, 2, 1).contiguous()  # (B, hidden_dim, L)
        y_bn = self.bn(y_perm)               # (B, hidden_dim, L)

        # back to (B, L, hidden_dim)
        y = y_bn.permute(0, 2, 1).contiguous()    # (B, L, hidden_dim)

        # 4: project to out_dim per position
        y = self.fc2(y)                      # (B, L, out_dim)

        # 5: global mean pooling over sequence dimension
        pooled = y.mean(dim=1)               # (B, out_dim)

        # 6: gating pathway: LeakyReLU then linear projection
        gated = self.leaky(pooled)           # (B, out_dim)
        gated = self.gate_proj(gated)        # (B, out_dim)

        # 7: compute gate and apply per-position (broadcast on seq dim)
        gate = torch.sigmoid(gated).unsqueeze(1)  # (B, 1, out_dim)
        out = y * gate                        # (B, L, out_dim) elementwise gating

        return out

# Configuration / example sizes
batch_size = 8
seq_len = 128
in_dim = 256
hidden_dim = 512
out_dim = 128
leaky_negative_slope = 0.2

def get_inputs():
    """
    Returns a list with a single randomized input tensor matching the model's expected input.
    Shape: (batch_size, seq_len, in_dim)
    """
    x = torch.randn(batch_size, seq_len, in_dim, dtype=torch.float32)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for the Model constructor.
    """
    return [in_dim, hidden_dim, out_dim, leaky_negative_slope]