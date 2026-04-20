import torch
import torch.nn as nn

# Configuration
batch_size = 8
channels = 64
length = 1024  # original 1D length
pad = 4        # ZeroPad1d padding on both sides
H = 8          # height to reshape into (must divide length + 2*pad)
out_dim = 512  # final output dimensionality

class Model(nn.Module):
    """
    Complex model that:
    - Pads a 1D sequence (ZeroPad1d)
    - Reshapes the padded sequence into a 2D spatial tensor (batch, C, H, W)
    - Upsamples spatially using nearest-neighbor upsampling (UpsamplingNearest2d)
    - Applies a channel-wise PReLU nonlinearity (nn.PReLU)
    - Reduces spatial dimensions by averaging and projects to out_dim via a Linear layer
    """
    def __init__(self, channels: int = channels, pad: int = pad, H: int = H, out_dim: int = out_dim):
        super(Model, self).__init__()
        self.channels = channels
        self.pad = pad
        self.H = H

        # Nearest-neighbor upsampling with different scale factors for height and width
        self.upsample = nn.UpsamplingNearest2d(scale_factor=(2, 3))

        # Channel-wise learnable PReLU (one parameter per channel)
        self.prelu = nn.PReLU(num_parameters=channels)

        # Final linear projection from channel-dimension to out_dim
        self.fc = nn.Linear(channels, out_dim)

        # ZeroPad1d layer for 1D padding
        self.pad1d = nn.ZeroPad1d(pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, channels, length)

        Returns:
            Tensor of shape (batch, out_dim)
        """
        # 1) Zero-pad the 1D sequence
        x = self.pad1d(x)  # -> (batch, channels, length + 2*pad)

        # 2) Reshape into 2D spatial tensor
        # Compute width after padding to ensure correct reshape
        length_padded = x.size(2)
        assert length_padded % self.H == 0, "Padded length must be divisible by H"
        W = length_padded // self.H
        x = x.view(x.size(0), self.channels, self.H, W)  # -> (batch, C, H, W)

        # 3) Upsample spatially (nearest neighbor)
        x = self.upsample(x)  # -> (batch, C, H*2, W*3)

        # 4) Apply channel-wise PReLU activation
        x = self.prelu(x)

        # 5) Spatial reduction: global average over H and W -> (batch, C)
        x = x.mean(dim=(2, 3))

        # 6) Final linear projection to desired output dimension
        out = self.fc(x)  # -> (batch, out_dim)

        return out

def get_inputs():
    """
    Generates a random input tensor compatible with the model.

    Returns:
        List containing one tensor of shape (batch_size, channels, length)
    """
    x = torch.randn(batch_size, channels, length)
    return [x]

def get_init_inputs():
    """
    No special initialization parameters required for this model.
    """
    return []