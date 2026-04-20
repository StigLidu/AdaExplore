import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex 3D processing module that:
    - Applies 1D replication padding along the last spatial dimension (W) by reshaping.
    - Applies InstanceNorm3d across channels.
    - Computes a channel-wise global average and adds it back as a residual.
    - Applies a ReLU activation.

    This creates a spatial-aware normalized residual block that pads only the W dimension
    while preserving the 3D structure for InstanceNorm3d.
    """
    def __init__(self, num_channels: int, pad_left: int, pad_right: int, eps: float = 1e-5, affine: bool = True):
        """
        Args:
            num_channels (int): Number of channels in the input tensor (C).
            pad_left (int): Amount of replication padding to add to the left of W.
            pad_right (int): Amount of replication padding to add to the right of W.
            eps (float): Epsilon value for InstanceNorm3d.
            affine (bool): Whether InstanceNorm3d has learnable affine params.
        """
        super(Model, self).__init__()
        # ReplicationPad1d expects (N, C, L). We'll reshape (N, C, D, H, W) -> (N*D*H, C, W) to pad W.
        self.pad1d = nn.ReplicationPad1d((pad_left, pad_right))
        self.inorm3d = nn.InstanceNorm3d(num_features=num_channels, eps=eps, affine=affine)
        self.relu = nn.ReLU()
        self.num_channels = num_channels
        self.pad_left = pad_left
        self.pad_right = pad_right

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (N, C, D, H, W_padded) after padding,
                          normalization, residual addition, and ReLU.
        """
        # Expecting a 5D tensor
        if x.dim() != 5:
            raise ValueError("Input tensor must be 5D (N, C, D, H, W)")

        N, C, D, H, W = x.shape

        # Permute to bring channels before W and combine spatial dims to use ReplicationPad1d
        # x_perm: (N, D, H, C, W)
        x_perm = x.permute(0, 2, 3, 1, 4)
        # Reshape to (N*D*H, C, W) so ReplicationPad1d can pad the W dimension
        x_reshaped = x_perm.reshape(N * D * H, C, W)

        # Apply 1D replication padding on W
        x_padded_reshaped = self.pad1d(x_reshaped)  # shape: (N*D*H, C, W_padded)
        W_padded = x_padded_reshaped.size(-1)

        # Restore to 5D shape: (N, C, D, H, W_padded)
        x_unreshaped = x_padded_reshaped.reshape(N, D, H, C, W_padded).permute(0, 3, 1, 2, 4)

        # Instance normalization across channels for the 3D volume
        x_norm = self.inorm3d(x_unreshaped)

        # Compute channel-wise global average over spatial dims (D, H, W_padded)
        gap = x_norm.mean(dim=(2, 3, 4), keepdim=True)  # shape: (N, C, 1, 1, 1)

        # Add the global average back as a residual (broadcast) and apply ReLU
        out = self.relu(x_norm + gap)

        return out

# Module-level configuration
batch_size = 4
channels = 8
depth = 6
height = 5
width = 10
pad_left = 2
pad_right = 3

def get_inputs():
    """
    Returns a list containing the input tensor for the model.
    Shape: (batch_size, channels, depth, height, width)
    """
    x = torch.randn(batch_size, channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters for the Model constructor:
    [num_channels, pad_left, pad_right]
    """
    return [channels, pad_left, pad_right]