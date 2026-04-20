import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Spatiotemporal feature extractor that:
      - Applies a 2D convolution to each temporal slice independently,
      - Reassembles those per-slice features into a 3D tensor,
      - Applies 3D max-pooling across time and spatial dims,
      - Collapses temporal dimension into channels and projects with a LazyLinear.

    This pattern demonstrates combining Conv2d, MaxPool3d and LazyLinear into a fused
    spatiotemporal pipeline.
    """
    def __init__(self, in_channels: int, conv_out_channels: int, fc_out_features: int):
        """
        Initializes the spatiotemporal model.

        Args:
            in_channels (int): Number of input channels per frame (e.g., RGB -> 3).
            conv_out_channels (int): Number of output channels produced by the 2D conv.
            fc_out_features (int): Output features for the final LazyLinear layer.
        """
        super(Model, self).__init__()
        # 2D conv applied to each temporal slice (per-frame spatial processing)
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=conv_out_channels, kernel_size=3, padding=1)
        # 3D max-pooling to reduce temporal + spatial resolution jointly
        self.pool3d = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # LazyLinear will infer in_features at first forward pass
        self.fc = nn.LazyLinear(out_features=fc_out_features)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, depth, height, width)

        Returns:
            torch.Tensor: Tensor of shape (batch_size, fc_out_features) after projection.
        """
        # x: (N, C, D, H, W)
        N, C, D, H, W = x.shape

        # Treat each temporal slice as an independent image: (N*D, C, H, W)
        x_per_slice = x.permute(0, 2, 1, 3, 4).contiguous().view(N * D, C, H, W)

        # Apply 2D conv to each slice
        conv_out = self.conv2d(x_per_slice)  # (N*D, conv_out_channels, H, W)

        # Reconstruct spatiotemporal tensor: (N, conv_out_channels, D, H, W)
        conv_out = conv_out.view(N, D, conv_out.shape[1], conv_out.shape[2], conv_out.shape[3])
        conv_out = conv_out.permute(0, 2, 1, 3, 4).contiguous()

        # 3D max pooling reduces temporal and spatial axes
        pooled = self.pool3d(conv_out)  # (N, conv_out_channels, D', H', W')

        # Global average over spatial dimensions (H', W') -> (N, conv_out_channels, D')
        pooled = pooled.mean(dim=[3, 4])

        # Collapse temporal (depth) into channel dimension: (N, conv_out_channels * D')
        N_out, ch, D_prime = pooled.shape
        flattened = pooled.view(N_out, ch * D_prime)

        # Final projection via LazyLinear (infers in_features on first call)
        out = self.fc(flattened)
        out = self.act(out)
        return out

# Configuration / default sizes
batch_size = 4
channels = 3
depth = 8
height = 64
width = 64
conv_out_channels = 16
fc_out_features = 128

def get_inputs():
    """
    Returns a list containing a single input tensor with shape:
    (batch_size, channels, depth, height, width)
    """
    x = torch.randn(batch_size, channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for Model:
    [in_channels, conv_out_channels, fc_out_features]
    """
    return [channels, conv_out_channels, fc_out_features]