import torch
import torch.nn as nn
import torch.nn.functional as F

# Configuration
batch_size = 8
in_channels = 3
height = 64
width = 64
kernel_size = 5
stride = 4
hidden_dim = 256
out_dim = 10

class Model(nn.Module):
    """
    Patch-based feature extractor that:
      - Extracts sliding patches with nn.Unfold
      - Projects each patch into a latent space (nn.Linear)
      - Applies SyncBatchNorm over the latent channels
      - Applies Hardsigmoid non-linearity
      - Computes attention weights over patches and aggregates
      - Produces a final classification/regression vector

    Input:
      x: Tensor of shape (B, C, H, W)

    Output:
      Tensor of shape (B, out_dim)
    """
    def __init__(self,
                 in_channels: int,
                 kernel_size: int,
                 stride: int,
                 hidden_dim: int,
                 out_dim: int):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        # Extract patches
        self.unfold = nn.Unfold(kernel_size=kernel_size, stride=stride)

        # Linear projection from flattened patch dimension to hidden_dim
        patch_dim = in_channels * kernel_size * kernel_size
        self.proj = nn.Linear(patch_dim, hidden_dim, bias=True)

        # Batch-norm applied across channels of the projected patches
        # SyncBatchNorm behaves like BatchNorm in single-process mode
        self.bn = nn.SyncBatchNorm(hidden_dim)

        # Non-linearity
        self.act = nn.Hardsigmoid()

        # Attention scorer for each patch (reduces hidden_dim -> 1 per patch)
        self.attn = nn.Linear(hidden_dim, 1, bias=True)

        # Final output head
        self.fc_out = nn.Linear(hidden_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation:
          1. Extract patches -> (B, patch_dim, L)
          2. Transpose -> (B, L, patch_dim)
          3. Project -> (B, L, hidden_dim)
          4. Apply SyncBatchNorm across hidden channels -> (B, L, hidden_dim)
          5. Apply Hardsigmoid -> (B, L, hidden_dim)
          6. Compute attention scores & softmax over L -> (B, L, 1)
          7. Weighted sum over patches -> (B, hidden_dim)
          8. Final linear head -> (B, out_dim)
        """
        # x: (B, C, H, W)
        patches = self.unfold(x)                       # (B, patch_dim, L)
        patches = patches.transpose(1, 2)              # (B, L, patch_dim)

        projected = self.proj(patches)                 # (B, L, hidden_dim)

        # SyncBatchNorm expects (B, C, L...) so move hidden_dim to channel dim
        projected_t = projected.transpose(1, 2)        # (B, hidden_dim, L)
        normalized = self.bn(projected_t)              # (B, hidden_dim, L)
        normalized = normalized.transpose(1, 2)        # (B, L, hidden_dim)

        activated = self.act(normalized)               # (B, L, hidden_dim)

        # Attention: compute a score per patch and softmax across patches
        scores = self.attn(activated).squeeze(-1)      # (B, L)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, L, 1)

        # Weighted aggregation over patches
        aggregated = torch.sum(activated * weights, dim=1)    # (B, hidden_dim)

        out = self.fc_out(aggregated)                   # (B, out_dim)
        return out

def get_inputs():
    """
    Returns example input tensors for the Model.
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters matching Model.__init__ signature.
    Order: in_channels, kernel_size, stride, hidden_dim, out_dim
    """
    return [in_channels, kernel_size, stride, hidden_dim, out_dim]