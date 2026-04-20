import torch
import torch.nn as nn

# Configuration
batch_size = 8
in_channels = 3
depth = 10
height = 32
width = 32
hidden_dim = 128
out_dim = 10

class Model(nn.Module):
    """
    A moderately-complex 3D-feature aggregation module that:
    - Applies replication padding in 3D to preserve boundary values
    - Computes both mean and max summaries over spatial dimensions
    - Concatenates these summaries and projects them with a LazyLinear layer
    - Applies a nonlinearity, a final linear projection, and a Softmax to produce probabilities

    Input shape: (batch_size, in_channels, depth, height, width)
    Output shape: (batch_size, out_dim) -- probabilities over out_dim classes
    """
    def __init__(self, in_channels: int, hidden_dim: int, out_dim: int):
        super(Model, self).__init__()
        # ReplicationPad3d padding tuple: (padW_left, padW_right, padH_top, padH_bottom, padD_front, padD_back)
        # We add a small pad in H and W to allow boundary replication; no pad in depth here.
        self.pad = nn.ReplicationPad3d((1, 1, 1, 1, 0, 0))
        # LazyLinear will infer in_features at first forward call based on concatenated summaries (in_channels * 2)
        self.fc1 = nn.LazyLinear(hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        # Softmax over class dimension
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        1. Replication pad the input in spatial H and W dimensions.
        2. Compute mean and max across spatial dims (D, H, W) to get two (batch, channels) summaries.
        3. Concatenate summaries to form (batch, channels*2).
        4. Project with a LazyLinear (inferred), apply ReLU, then final linear and Softmax.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, D, H, W)

        Returns:
            torch.Tensor: Output probabilities of shape (N, out_dim)
        """
        # 1) Pad
        x_padded = self.pad(x)  # -> (N, C, D, H+2, W+2)
        # 2) Spatial summaries
        spatial_mean = x_padded.mean(dim=(2, 3, 4))  # (N, C)
        spatial_max = x_padded.amax(dim=(2, 3, 4))   # (N, C)
        # 3) Concatenate summaries
        combined = torch.cat([spatial_mean, spatial_max], dim=1)  # (N, C*2)
        # 4) Project, activate, final project, softmax
        hidden = self.fc1(combined)  # LazyLinear infers in_features here
        hidden = self.act(hidden)
        logits = self.fc2(hidden)
        probs = self.softmax(logits)
        return probs

def get_inputs():
    """
    Creates a random 5D input tensor matching the expected shape:
    (batch_size, in_channels, depth, height, width)
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters required to construct the Model.
    """
    return [in_channels, hidden_dim, out_dim]