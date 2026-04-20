import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Patch-aware channel gating model.

    This model demonstrates a combination of nn.Unfold to extract local patches,
    a learned linear projection on each patch to produce channel logits,
    nn.LogSoftmax to produce stabilized log-probabilities over channels for each patch,
    and nn.Dropout3d to randomly drop entire channels before applying a
    channel-wise gating computed from the patch statistics.

    Forward pass summary:
    1. Extract patches from input using Unfold -> (B, C * K * K, L)
    2. Project each patch vector to per-channel logits -> (B, L, C)
    3. Convert logits to log-probabilities over channels (LogSoftmax)
    4. Average the per-patch probabilities to obtain per-channel scores (B, C)
    5. Apply Dropout3d to the original input and multiply channel-wise by the scores
       (broadcasted over spatial dims) producing gated output of shape (B, C, H, W).
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int, dropout_p: float = 0.2):
        """
        Args:
            in_channels (int): Number of channels in the input tensor.
            kernel_size (int): Size of the square patch (K).
            stride (int): Stride when extracting patches.
            dropout_p (float): Probability for Dropout3d.
        """
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride

        # Extract sliding local blocks (patches)
        self.unfold = nn.Unfold(kernel_size=kernel_size, stride=stride)

        # Map each patch (C * K * K) -> per-channel logits (C)
        patch_dim = in_channels * kernel_size * kernel_size
        self.proj = nn.Linear(patch_dim, in_channels, bias=True)

        # Stabilized log-probabilities across channel dimension for each patch
        # We'll use dim=2 because projected shape will be (B, L, C)
        self.logsoftmax = nn.LogSoftmax(dim=2)

        # Dropout that zeros entire channels (3D dropout expects (N, C, D, H, W) or (N, C, H, W))
        self.dropout3d = nn.Dropout3d(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Gated tensor of same shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        # 1) Extract patches: (B, C * K * K, L)
        patches = self.unfold(x)  # shape: (B, patch_dim, L)

        # 2) Rearrange to (B, L, patch_dim) to process patches as a sequence
        patches = patches.permute(0, 2, 1)  # (B, L, patch_dim)
        B_, L, patch_dim = patches.shape  # B_ == B

        # 3) Project each patch to per-channel logits using a linear layer
        patches_flat = patches.reshape(B_ * L, patch_dim)  # (B*L, patch_dim)
        logits = self.proj(patches_flat)  # (B*L, C)
        logits = logits.view(B_, L, C)  # (B, L, C)

        # 4) Convert logits to log-probabilities over channels for each patch
        log_probs = self.logsoftmax(logits)  # (B, L, C)

        # 5) Convert to probabilities and pool across patches to obtain per-channel scores
        probs = torch.exp(log_probs)  # (B, L, C)
        channel_scores = probs.mean(dim=1)  # (B, C)  -- average probability per channel

        # 6) Reshape scores to broadcast over spatial dims: (B, C, 1, 1)
        channel_scores = channel_scores.view(B, C, 1, 1)

        # 7) Apply Dropout3d to the original input (channel-wise dropout)
        x_dropped = self.dropout3d(x)

        # 8) Apply channel-wise gating
        gated = x_dropped * channel_scores  # (B, C, H, W)

        return gated

# Configuration variables
batch_size = 8
in_channels = 16
height = 64
width = 48
kernel_size = 3
stride = 2
dropout_p = 0.25

def get_inputs():
    """
    Returns:
        list: Single input tensor with shape (batch_size, in_channels, height, width)
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for Model constructor:
        [in_channels, kernel_size, stride, dropout_p]
    """
    return [in_channels, kernel_size, stride, dropout_p]