import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex model that:
    - Applies a learned linear mixing across feature dimension for each timestep.
    - Normalizes across groups of channels using GroupNorm.
    - Applies a HardTanh activation.
    - Pools over the temporal dimension (max pooling).
    - Applies a lazily-initialized BatchNorm1d over the feature dimension.
    - Projects to a scalar per example and applies a sigmoid.

    Input shape: (batch_size, seq_len, in_features)
    Output shape: (batch_size, 1)
    """
    def __init__(self, in_features: int, hidden_features: int, num_groups: int):
        super(Model, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.num_groups = num_groups

        # Learned linear mixing across the feature axis: X @ weight + bias
        self.weight = nn.Parameter(torch.randn(in_features, hidden_features) * (1.0 / (in_features ** 0.5)))
        self.bias = nn.Parameter(torch.zeros(hidden_features))

        # Group normalization across the "channel" (hidden_features) dimension.
        # GroupNorm expects num_channels to be divisible by num_groups.
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_features)

        # HardTanh non-linearity.
        self.hardtanh = nn.Hardtanh(min_val=-1.0, max_val=1.0)

        # Lazy BatchNorm1d will be initialized on the first forward pass when feature dim is known.
        self.lazy_bn = nn.LazyBatchNorm1d()

        # Final projection to a single scalar per example.
        self.output_vector = nn.Parameter(torch.randn(hidden_features, 1) * (1.0 / (hidden_features ** 0.5)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, S, F) where
               B = batch size, S = sequence length, F = in_features.

        Returns:
            Tensor of shape (B, 1) with values in (0,1) after sigmoid.
        """
        # Linear mixing across feature dimension for each timestep:
        # (B, S, F) @ (F, H) -> (B, S, H)
        x = torch.matmul(x, self.weight) + self.bias

        # Reorder to (B, H, S) to apply GroupNorm treating H as channels.
        x = x.permute(0, 2, 1)

        # Group normalization (operates over channel dimension).
        x = self.groupnorm(x)

        # Non-linearity
        x = self.hardtanh(x)

        # Pool across the temporal/sequence dimension: max over last dim -> (B, H)
        x = torch.max(x, dim=2)[0]

        # Lazy BatchNorm1d will initialize with num_features = H on first call.
        x = self.lazy_bn(x)

        # Final projection to scalar per example: (B, H) @ (H, 1) -> (B, 1)
        out = torch.matmul(x, self.output_vector)

        # Sigmoid to bound outputs between 0 and 1.
        return torch.sigmoid(out)


# Configuration variables
batch_size = 32
seq_len = 128
in_features = 512
hidden_features = 1024
num_groups = 16  # hidden_features must be divisible by num_groups (1024 % 16 == 0)

def get_inputs():
    """
    Returns a list containing a single input tensor shaped (batch_size, seq_len, in_features).
    """
    x = torch.randn(batch_size, seq_len, in_features)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for constructing the Model: (in_features, hidden_features, num_groups).
    """
    return [in_features, hidden_features, num_groups]