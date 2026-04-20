import torch
import torch.nn as nn
from typing import List, Tuple

class Model(nn.Module):
    """
    Complex model combining 3D Lp pooling, sequence processing with a GRU,
    and sparsity-inducing Hardshrink activation followed by gating and projection.

    Overall computation:
    1. LPPool3d over spatial dimensions (D, H, W).
    2. Rearrange pooled features into a sequence (seq_len = D'*H'*W') with feature dim = C.
    3. Project per-step features into GRU input space.
    4. Process sequence with a multi-layer GRU (batch_first=True).
    5. Apply Hardshrink to GRU outputs to induce sparsity.
    6. Compute a sigmoid gate from the shrunk outputs and elementwise-multiply (gating).
    7. Temporal average pooling across the sequence and final linear projection to outputs.
    """
    def __init__(
        self,
        in_channels: int,
        lp_norm: float,
        kernel_size: Tuple[int, int, int],
        stride: Tuple[int, int, int],
        gru_input_size: int,
        gru_hidden: int,
        gru_layers: int,
        shrink_lambda: float,
        out_features: int,
    ):
        super(Model, self).__init__()
        # 3D Lp pooling
        self.pool = nn.LPPool3d(norm_type=lp_norm, kernel_size=kernel_size, stride=stride)

        # Project C -> gru_input_size for each timestep
        self.input_proj = nn.Linear(in_channels, gru_input_size)

        # GRU operates on the projected sequence
        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=False
        )

        # Sparsity-inducing activation
        self.hardshrink = nn.Hardshrink(lambd=shrink_lambda)

        # Simple gating mechanism (maps shrunk features to [0,1] per feature)
        self.gate_linear = nn.Linear(gru_hidden, gru_hidden)

        # Final projection after temporal pooling
        self.output_linear = nn.Linear(gru_hidden, out_features)

        # Keep some shapes for potential debugging/inspection
        self.in_channels = in_channels
        self.gru_input_size = gru_input_size
        self.gru_hidden = gru_hidden
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, D, H, W)

        Returns:
            Tensor of shape (B, out_features)
        """
        # 1) Lp pooling -> (B, C, D', H', W')
        pooled = self.pool(x)

        # 2) Rearrange into sequence: (B, seq_len, C) where seq_len = D'*H'*W'
        B, C, Dp, Hp, Wp = pooled.shape
        seq_len = Dp * Hp * Wp
        seq = pooled.permute(0, 2, 3, 4, 1).reshape(B, seq_len, C)

        # 3) Project per-timestep features -> (B, seq_len, gru_input_size)
        proj = self.input_proj(seq)

        # 4) GRU sequence processing -> (B, seq_len, gru_hidden)
        gru_out, _ = self.gru(proj)

        # 5) Hardshrink encourages sparsity -> (B, seq_len, gru_hidden)
        shrunk = self.hardshrink(gru_out)

        # 6) Gating: sigmoid(linear(shrunk)) * shrunk -> (B, seq_len, gru_hidden)
        gate = torch.sigmoid(self.gate_linear(shrunk))
        gated = shrunk * gate

        # 7) Temporal average pooling across sequence dimension -> (B, gru_hidden)
        pooled_feat = gated.mean(dim=1)

        # Final projection
        out = self.output_linear(pooled_feat)
        return out

# Configuration: sizes and hyperparameters
batch_size = 8
in_channels = 16
depth = 8
height = 16
width = 16

lp_norm = 2.0  # p value for Lp pooling
pool_kernel = (2, 2, 2)
pool_stride = (2, 2, 2)

gru_input_size = 64
gru_hidden = 128
gru_layers = 2

shrink_lambda = 0.5
out_features = 256

def get_inputs() -> List[torch.Tensor]:
    """
    Returns a list containing a single input tensor of shape (B, C, D, H, W).
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs() -> List:
    """
    Returns initialization parameters for the Model constructor in the correct order.
    """
    return [
        in_channels,
        lp_norm,
        pool_kernel,
        pool_stride,
        gru_input_size,
        gru_hidden,
        gru_layers,
        shrink_lambda,
        out_features,
    ]