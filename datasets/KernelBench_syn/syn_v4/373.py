import torch
import torch.nn as nn
from typing import List

class Model(nn.Module):
    """
    Complex 1D sequence processing module that combines:
    - a expanding Conv1d
    - Group Normalization
    - Softshrink non-linearity
    - Dropout1d (channel-wise dropout)
    - Adaptive average pooling along the temporal dimension
    - A linear projection applied per pooled time-step to produce final channel outputs

    The model demonstrates a compact but non-trivial computation graph with residual considerations
    and careful tensor reshaping to apply a Linear layer per time-step.
    """
    def __init__(
        self,
        in_channels: int,
        expansion: int = 2,
        group_norm_groups: int = 8,
        softshrink_lambda: float = 0.5,
        dropout_p: float = 0.2,
        pool_output_size: int = 128,
        out_channels: int = None,
    ):
        """
        Args:
            in_channels: Number of input channels/features per time step.
            expansion: Factor to expand channels after the first Conv1d.
            group_norm_groups: Number of groups for GroupNorm (will be clipped to valid range).
            softshrink_lambda: Lambda parameter for Softshrink activation.
            dropout_p: Probability for Dropout1d.
            pool_output_size: Temporal length after AdaptiveAvgPool1d.
            out_channels: Desired output channels. If None, defaults to in_channels.
        """
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.expanded_channels = in_channels * expansion
        self.pool_output_size = pool_output_size
        self.out_channels = out_channels if out_channels is not None else in_channels

        # 1. Expand channels with a 1D convolution (preserve temporal length with padding)
        self.conv_expand = nn.Conv1d(in_channels, self.expanded_channels, kernel_size=3, padding=1, bias=False)

        # 2. GroupNorm expects num_channels = expanded_channels. Ensure groups divides channels.
        # Clip group count to be at most expanded_channels and at least 1, and to divide expanded_channels when possible.
        groups = max(1, min(group_norm_groups, self.expanded_channels))
        # If groups does not divide expanded_channels, reduce groups until it does.
        while self.expanded_channels % groups != 0:
            groups -= 1
            if groups == 1:
                break
        self.gn = nn.GroupNorm(num_groups=groups, num_channels=self.expanded_channels)

        # 3. Softshrink non-linearity
        self.softshrink = nn.Softshrink(lambd=softshrink_lambda)

        # 4. Channel-wise dropout
        self.dropout = nn.Dropout1d(p=dropout_p)

        # 5. Temporal pooling to reduce sequence length
        self.pool = nn.AdaptiveAvgPool1d(self.pool_output_size)

        # 6. Linear projection applied per pooled time-step to map expanded_channels -> out_channels.
        # We'll apply it by reshaping to (batch * pool_len, channels) so Linear can be used directly.
        self.proj = nn.Linear(self.expanded_channels, self.out_channels, bias=True)

        # Optional small residual projection if out_channels != in_channels, to make output shape intuitive.
        if self.out_channels != self.in_channels:
            self.residual_proj = nn.Conv1d(self.in_channels, self.out_channels, kernel_size=1, bias=False)
        else:
            self.residual_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch_size, in_channels, seq_len)

        Returns:
            Tensor of shape (batch_size, out_channels, pool_output_size)
        """
        # Save residual (may be projected later)
        residual = x

        # 1. Expand channels
        out = self.conv_expand(x)  # (B, expanded_channels, L)

        # 2. Group Normalization
        out = self.gn(out)

        # 3. Non-linear shrinkage
        out = self.softshrink(out)

        # 4. Dropout across channels
        out = self.dropout(out)

        # 5. Temporal pooling
        out = self.pool(out)  # (B, expanded_channels, pool_output_size)

        # 6. Linear projection applied per time-step.
        # Rearrange to apply Linear on channel dimension for each time-step independently.
        B, C, T = out.shape
        # (B * T, C)
        out_flat = out.permute(0, 2, 1).reshape(B * T, C)
        out_proj = self.proj(out_flat)  # (B * T, out_channels)
        # Back to (B, out_channels, T)
        out = out_proj.reshape(B, T, self.out_channels).permute(0, 2, 1)

        # If residual projection exists, project and pool the residual to match temporal length.
        # Use adaptive avg pool on residual's temporal dimension to match pool_output_size.
        if self.residual_proj is not None:
            res = self.residual_proj(residual)  # (B, out_channels, L)
        else:
            res = residual  # (B, in_channels, L) where in_channels == out_channels

        # Pool residual temporally to match output T
        res = nn.functional.adaptive_avg_pool1d(res, self.pool_output_size)  # (B, out_channels, T)

        # Fuse with residual (simple addition) and return
        return out + res


# Module-level configuration variables (can be used by get_inputs / get_init_inputs)
batch_size = 8
in_channels = 32
sequence_length = 512
expansion = 2
group_norm_groups = 8
softshrink_lambda = 0.5
dropout_p = 0.2
pool_output_size = 128
out_channels = 48  # intentionally different from in_channels to exercise residual projection

def get_inputs() -> List[torch.Tensor]:
    """
    Returns a list containing a single input tensor shaped (batch_size, in_channels, sequence_length).
    """
    x = torch.randn(batch_size, in_channels, sequence_length)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters required to construct the Model in the following order:
    [in_channels, expansion, group_norm_groups, softshrink_lambda, dropout_p, pool_output_size, out_channels]
    """
    return [in_channels, expansion, group_norm_groups, softshrink_lambda, dropout_p, pool_output_size, out_channels]