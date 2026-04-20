import torch
import torch.nn as nn

"""
Complex example module combining ZeroPad1d, RMSNorm, and Softshrink with a learned
1D "filtering" operation implemented via tensor.unfold and a final projection.

Structure:
- Model(nn.Module)
  - __init__(in_channels, out_channels, kernel_size, pad, shrink_lambda, eps)
  - forward(x: Tensor) -> Tensor
- get_inputs() -> list of input tensors
- get_init_inputs() -> list of initialization arguments for Model

Computation steps in forward:
1. Expect input x shaped (batch, seq_len, channels).
2. Permute to (batch, channels, seq_len).
3. Apply ZeroPad1d to pad temporal dimension.
4. Use Tensor.unfold to create sliding windows over time.
5. Multiply windows by a learnable kernel per channel and sum over the window (1D filtering).
6. Permute to (batch, new_seq_len, channels).
7. Apply RMSNorm across the channel dimension.
8. Apply Softshrink non-linearity to induce sparsity.
9. Apply a Linear projection (per time step) to map channels -> out_channels.
10. Aggregate across time (mean) to produce (batch, out_channels) output.
"""

# Configuration / default sizes
BATCH = 8
SEQ_LEN = 512
IN_CHANNELS = 32
OUT_CHANNELS = 64
KERNEL_SIZE = 7
PAD = 3  # symmetric padding on both sides
SHRINK_LAMBDA = 0.25
RMS_EPS = 1e-6

class Model(nn.Module):
    """
    Model combining padding, sliding-window filtering, RMSNorm, Softshrink, and projection.

    Args:
        in_channels (int): Number of input channels/features.
        out_channels (int): Number of output features after projection.
        kernel_size (int): Size of the temporal window for the learned filter.
        pad (int): Amount of zero-padding applied to both ends of the temporal dimension.
        shrink_lambda (float): Lambda parameter for Softshrink.
        eps (float): Epsilon for RMSNorm stability.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pad: int,
        shrink_lambda: float = 0.5,
        eps: float = 1e-5,
    ):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pad = pad
        self.shrink_lambda = shrink_lambda
        self.eps = eps

        # Pads the temporal (last) dimension for tensors shaped (N, C, L)
        # Using symmetric padding (pad_left = pad_right)
        self.pad_layer = nn.ZeroPad1d((pad, pad))

        # RMSNorm normalizes over the last dimension; we will apply it over channels
        # after permuting to (batch, seq, channels).
        self.rmsnorm = nn.RMSNorm(in_channels, eps=self.eps)

        # Softshrink non-linearity for sparsity
        self.softshrink = nn.Softshrink(lambd=self.shrink_lambda)

        # Learnable kernel per input channel for the sliding-window filtering
        # Shape: (in_channels, kernel_size)
        self.kernel = nn.Parameter(torch.randn(in_channels, kernel_size) * (1.0 / kernel_size))

        # Linear projection applied to the channel dimension at each time-step:
        # maps in_channels -> out_channels
        self.proj = nn.Linear(in_channels, out_channels, bias=True)

        # Initialize projection and kernel in a stable manner
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
        nn.init.normal_(self.kernel, mean=0.0, std=(1.0 / (kernel_size ** 0.5)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, in_channels).

        Returns:
            torch.Tensor: Output tensor of shape (batch, out_channels).
        """
        # Expect input (B, S, C)
        if x.dim() != 3:
            raise ValueError(f"Expected input of shape (B, S, C), got {tuple(x.shape)}")

        # 1) Permute to (B, C, S) for padding and unfold
        x = x.permute(0, 2, 1)  # (B, C, S)

        # 2) Zero-pad temporal dimension -> (B, C, S + 2*pad)
        x_padded = self.pad_layer(x)

        # 3) Create sliding windows over temporal dimension using Tensor.unfold
        # windows shape: (B, C, L_w, kernel_size)
        windows = x_padded.unfold(dimension=2, size=self.kernel_size, step=1)

        # 4) Apply per-channel learned kernel weights and sum over kernel dimension
        # Expand kernel to (1, C, 1, K) to broadcast over batch and time positions
        kernel_expanded = self.kernel.unsqueeze(0).unsqueeze(2)  # (1, C, 1, K)
        filtered = (windows * kernel_expanded).sum(dim=-1)  # (B, C, L_w)

        # 5) Permute to (B, L_w, C) to apply RMSNorm and Linear which operate on last dim
        filtered = filtered.permute(0, 2, 1)  # (B, L_w, C)

        # 6) RMS normalization across channel dimension
        normalized = self.rmsnorm(filtered)  # (B, L_w, C)

        # 7) Non-linear sparsification
        sparse = self.softshrink(normalized)  # (B, L_w, C)

        # 8) Per-time-step projection to out_channels
        projected = self.proj(sparse)  # (B, L_w, out_channels)

        # 9) Temporal aggregation: mean over time dimension -> (B, out_channels)
        out = projected.mean(dim=1)

        return out

def get_inputs():
    """
    Returns example input tensors for the model:
    - x: (BATCH, SEQ_LEN, IN_CHANNELS)
    """
    x = torch.randn(BATCH, SEQ_LEN, IN_CHANNELS)
    return [x]

def get_init_inputs():
    """
    Returns initialization arguments for Model in the same order as the constructor.
    """
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, PAD, SHRINK_LAMBDA, RMS_EPS]