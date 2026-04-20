import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex module that combines Softmax2d attention over channels per spatial
    location, a channel-wise summary with Softsign non-linearity, channel dropout
    (Dropout1d), and re-scaling (gating) of the original feature maps.

    Computation steps:
    1. Compute per-location channel softmax: s = Softmax2d(x)
    2. Apply attention: weighted = x * s
    3. Channel summary: summary = mean(weighted, dim=(2,3))  # shape (N, C)
    4. Non-linearity: shrunk = Softsign(summary)
    5. Channel dropout: dropped = Dropout1d(shrunk.unsqueeze(-1)).squeeze(-1)
    6. Re-scale spatial feature maps by expanded channel gates and return pooled output
       final = mean(weighted * gates, dim=(2,3))  # shape (N, C)
    """
    def __init__(self, dropout_p: float = 0.2):
        """
        Initializes the composite module.

        Args:
            dropout_p (float): Probability of dropping a channel in Dropout1d.
        """
        super(Model, self).__init__()
        # Softmax2d applies softmax over channel dimension for every (H, W)
        self.softmax2d = nn.Softmax2d()
        # Softsign is an elementwise bounded non-linearity
        self.softsign = nn.Softsign()
        # Dropout1d zeros out entire channels (operates on (N, C, L))
        self.dropout1d = nn.Dropout1d(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, height, width).

        Returns:
            torch.Tensor: Channel-wise pooled output after attention and gating,
                          shape (batch, channels).
        """
        # 1) Per-spatial-location softmax across channels -> attention map
        attn = self.softmax2d(x)  # (N, C, H, W)

        # 2) Apply attention to the features
        weighted = x * attn  # (N, C, H, W)

        # 3) Channel-wise summary via spatial average
        # result shape: (N, C)
        summary = weighted.mean(dim=(2, 3))

        # 4) Non-linear shrinkage to keep gating values bounded
        gated = self.softsign(summary)  # (N, C)

        # 5) Channel dropout: Dropout1d expects (N, C, L) -> use L=1
        gated_expanded = gated.unsqueeze(-1)  # (N, C, 1)
        gated_dropped = self.dropout1d(gated_expanded).squeeze(-1)  # (N, C)

        # 6) Re-scale the spatial feature maps by the (possibly dropped) channel gates
        gates = gated_dropped.unsqueeze(-1).unsqueeze(-1)  # (N, C, 1, 1)
        reweighted = weighted * gates  # (N, C, H, W)

        # Final pooling to produce a compact representation (per-channel)
        out = reweighted.mean(dim=(2, 3))  # (N, C)
        return out


# Configuration variables
batch_size = 8
channels = 128
height = 32
width = 32
dropout_p = 0.25

def get_inputs():
    """
    Returns a list with a single input tensor to the model:
    shape (batch_size, channels, height, width)
    """
    x = torch.randn(batch_size, channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for the Model constructor.
    In this case, the dropout probability.
    """
    return [dropout_p]