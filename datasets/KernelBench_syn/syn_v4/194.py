import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex 3D processing model that demonstrates a mixture of lazy transposed convolution,
    adaptive pooling, 1D max-pooling/unpooling over flattened spatial dimensions, and channel
    reduction via 1x1 convolution. The model ends with a global adaptive pooling and a linear head.
    """
    def __init__(self, num_classes: int = 10):
        """
        Initializes the Model.

        Args:
            num_classes (int): Number of output classes for the final linear layer.
        """
        super(Model, self).__init__()
        # Lazily infer in_channels from the first forward pass
        # This transposed convolution upsamples spatial dims by stride=2 (approx).
        self.deconv = nn.LazyConvTranspose3d(out_channels=32, kernel_size=3, stride=2, padding=1)

        # Reduce/normalize spatial volume to a fixed small 3D grid
        self.adaptive_pool = nn.AdaptiveMaxPool3d((4, 4, 4))

        # 1D pooling/unpooling over flattened spatial dimensions (sequence length = 4*4*4 = 64)
        # MaxPool1d with return_indices to be invertible by MaxUnpool1d
        self.pool1d = nn.MaxPool1d(kernel_size=2, stride=2, return_indices=True)
        self.unpool1d = nn.MaxUnpool1d(kernel_size=2, stride=2)

        # After concatenating the original pooled features and the unpooled reconstruction
        # we reduce channels with a 1x1x1 convolution.
        # in_channels for this conv will be 32 (deconv out) * 2 after concat.
        self.reduce_conv = nn.Conv3d(in_channels=32 * 2, out_channels=16, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)

        # Final global pooling and classification head
        self.global_pool = nn.AdaptiveMaxPool3d((1, 1, 1))
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Steps:
        1. Upsample using lazy ConvTranspose3d (in_channels inferred lazily).
        2. AdaptiveMaxPool3d to a fixed (4,4,4) grid.
        3. Flatten spatial dims into a 1D sequence and apply MaxPool1d (with indices).
        4. Reconstruct via MaxUnpool1d back to the flattened sequence length.
        5. Reshape back to 3D grid, concatenate with the original pooled grid (skip connection).
        6. Reduce channels with a 1x1x1 conv, apply ReLU.
        7. Global adaptive pooling and final linear classification.

        Args:
            x (torch.Tensor): Input tensor with shape (N, C_in, D, H, W).

        Returns:
            torch.Tensor: Output tensor with shape (N, num_classes).
        """
        # 1) Transposed convolution (upsample)
        x_up = self.deconv(x)  # shape: (N, 32, D_up, H_up, W_up)

        # 2) Adaptive spatial pooling to fixed small grid
        x_small = self.adaptive_pool(x_up)  # shape: (N, 32, 4, 4, 4)

        N, C, D, H, W = x_small.shape

        # 3) Flatten spatial dims to a sequence and apply 1D max-pooling with indices
        x_flat = x_small.view(N, C, -1)  # shape: (N, C, L) where L = 64
        x_pooled, indices = self.pool1d(x_flat)  # x_pooled: (N, C, L/2)

        # 4) Unpool back to original flattened length
        x_unpooled = self.unpool1d(x_pooled, indices, output_size=x_flat.shape)  # (N, C, L)

        # 5) Reshape back to 3D grid
        x_unflat = x_unpooled.view(N, C, D, H, W)  # (N, 32, 4, 4, 4)

        # 6) Concatenate along channels (skip-like connection) and reduce channels
        x_comb = torch.cat([x_small, x_unflat], dim=1)  # (N, 64, 4, 4, 4)
        x_reduced = self.reduce_conv(x_comb)  # (N, 16, 4, 4, 4)
        x_act = self.relu(x_reduced)

        # 7) Global pooling and classification head
        x_glob = self.global_pool(x_act)  # (N, 16, 1, 1, 1)
        x_vec = x_glob.view(N, -1)  # (N, 16)
        out = self.fc(x_vec)  # (N, num_classes)

        return out

# Configuration variables
batch_size = 8
in_channels = 3
depth = 6
height = 8
width = 8
num_classes = 10

def get_inputs():
    """
    Returns a list containing a single input tensor that matches the expected input shape:
    (batch_size, in_channels, depth, height, width).
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization inputs if any are required externally. This model uses lazy initialization,
    but no external init parameters are needed for this harness.
    """
    return []  # No external initialization inputs required